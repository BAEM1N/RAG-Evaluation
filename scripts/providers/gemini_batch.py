#!/usr/bin/env python3
"""Gemini Batch API provider for RAG-Evaluation.

Submits and collects batches against the Google Gemini Batch API for three
workloads used across the evaluation pipeline:

1. **Generation** — Phase 5A candidate LLM answers (Gemini direct).
2. **Embedding** — ``gemini-embedding-001`` / ``text-multilingual-embedding-002``.
3. **Judge**     — flagship judge (``gemini-3.1-pro`` / ``gemini-3-pro-preview``
   legacy) scoring candidates against the reference answer.

Design rules (non-negotiable)
-----------------------------
* **No output-token cap.** We never set ``max_output_tokens`` / ``max_tokens`` —
  Gemini is allowed to run to natural completion inside its context window. The
  user has explicitly forbidden output caps for the whole pipeline.
* Batch API gives a 50% discount and is paired with **Context Caching** for the
  static judge prefix (rubric + question + reference) so that the 39 candidates
  per cell share one cache-write and 38 cache-reads.
* Auth: reads ``GOOGLE_API_KEY`` (falls back to ``GEMINI_API_KEY``). The
  ``google-genai>=1.73`` SDK picks up ``GOOGLE_API_KEY`` automatically.
* Cache minimums: Flash 1024 tok, Pro 4096 tok. We pad with a short Korean
  RAG judging system instruction when the static prefix is too short for Pro.
* TTL: caches are created with ``ttl='1h'`` (batch jobs finish within the hour
  in typical workloads; extend in caller if needed).
* Judge scoring extraction: last ``\\b[1-5]\\b`` token, else ``0`` fallback
  (mirrors ``scripts/llm_judge.py``).

API surface
-----------
See module-level functions ``write_generation_batch_jsonl``,
``write_embedding_batch_jsonl``, ``create_judge_cache``,
``write_judge_batch_jsonl_with_cache``, ``submit_batch``, ``poll_batch``,
``download_batch_output``, ``parse_generation_results``,
``parse_embedding_results``, ``parse_judge_results``.

The ``__main__`` section demonstrates the full cache→submit→poll→parse flow
with a tiny fake payload. It does **not** hit the network unless executed
with ``--live`` and a real API key.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# ── SDK import guarded so module is importable even without google-genai ──
try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    genai_types = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Rubrics — ported verbatim from scripts/llm_judge.py EVAL_PROMPTS so Gemini
# judges on identical criteria as the local llama.cpp / ollama judge.
# ─────────────────────────────────────────────────────────────────────────────
EVAL_PROMPTS: dict[str, str] = {
    "similarity": """You are an impartial judge rating answer similarity in Korean RAG.

Question: {question}
Reference (ground truth): {target}
Candidate answer: {generated}

Rate semantic similarity of the candidate to the reference on a 1-5 scale:
1 — Little to no semantic similarity; candidate does not convey the reference meaning.
2 — Partial similarity on some aspects, but major meaning diverges.
3 — Moderate similarity; core idea matches but notable differences.
4 — Candidate aligns with reference in most aspects with substantial similarity.
5 — Candidate closely aligns with the reference in all significant aspects.

Respond with ONLY the integer 1, 2, 3, 4, or 5.""",
    "correctness": """You are an impartial judge rating answer correctness in Korean RAG.

Question: {question}
Reference (ground truth): {target}
Candidate answer: {generated}

Rate factual correctness of the candidate against the reference on a 1-5 scale:
1 — Completely incorrect or contradicts the reference.
2 — Partial correctness but significant discrepancies (wrong numbers, wrong entities).
3 — Addresses several aspects accurately with some minor omissions or inaccuracies.
4 — Mostly correct with only minor omissions or inaccuracies.
5 — Correct with high accuracy and full alignment to the reference facts.

Respond with ONLY the integer 1, 2, 3, 4, or 5.""",
    "completeness": """You are an impartial judge rating answer completeness in Korean RAG.

Question: {question}
Reference (ground truth): {target}
Candidate answer: {generated}

Rate how completely the candidate covers the key points from the reference on a 1-5 scale:
1 — Very incomplete; most key points from the reference are missing.
2 — Only a small fraction of key points covered; major gaps remain.
3 — About half of the key points covered; notable omissions.
4 — Most key points covered; only minor details missing.
5 — Fully complete; every important element from the reference is present.

Respond with ONLY the integer 1, 2, 3, 4, or 5.""",
    "faithfulness": """You are an impartial judge rating answer faithfulness in Korean RAG.

Question: {question}
Reference (ground truth): {target}
Candidate answer: {generated}

Rate how faithful the candidate is — i.e., free from hallucinations or claims that contradict the reference — on a 1-5 scale:
1 — Many hallucinations or claims that contradict the reference.
2 — Several unsupported or contradictory claims alongside some valid content.
3 — Mixed; some unsupported claims but core content aligns with the reference.
4 — Mostly faithful; only one or two minor unsupported details.
5 — Fully faithful; every claim is supported by or consistent with the reference.

Respond with ONLY the integer 1, 2, 3, 4, or 5.""",
}

# Korean RAG judging system padding — used to push cache payload past the
# 4096-token minimum on gemini-3.1-pro when Q+reference+rubric is too short.
_PAD_SYSTEM = (
    "You are a meticulous bilingual (Korean/English) evaluator for retrieval-augmented "
    "generation. Korean texts may use honorifics, postpositions and mixed Hanja — treat "
    "semantically equivalent surface variants as equivalent. Numerical facts, named "
    "entities, dates and units MUST match the reference. If the candidate adds content "
    "that is not in the reference, penalize faithfulness and completeness accordingly. "
    "If the candidate is a refusal or says '모릅니다' while the reference has content, "
    "score 1 for correctness and completeness. Emit ONLY one integer 1-5 at the end of "
    "the response — no prose, no markdown, no JSON. This instruction is embedded in a "
    "reusable context cache so the same rubric applies to every candidate for this "
    "(question, metric) cell. Do not restate the reference or question in the output."
)

# Cache minimum tokens (from Gemini docs — 2026-Q1).
CACHE_MIN_TOKENS = {"flash": 1024, "pro": 4096}

# Default poll interval for batch jobs (seconds).
DEFAULT_POLL_INTERVAL = 60

# Terminal batch job states.
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

# Short generation system prompt — mirrors phase5_provider.RAG_PROMPT.
RAG_PROMPT = (
    "다음 검색된 문맥을 사용하여 질문에 답하세요. "
    "답을 모르면 모른다고 하세요. 최대 3문장으로 간결하게 답하세요.\n\n"
    "질문: {question}\n\n문맥:\n{context}\n\n답변:"
)


# ─────────────────────────────────────────────────────────────────────────────
# Client helper
# ─────────────────────────────────────────────────────────────────────────────
def _client() -> Any:
    """Instantiate a google-genai client using GOOGLE_API_KEY (or GEMINI_API_KEY).

    The SDK preferentially reads ``GOOGLE_API_KEY``. If only ``GEMINI_API_KEY``
    is set we copy it into ``GOOGLE_API_KEY`` for this process.
    """
    if genai is None:
        raise RuntimeError(
            "google-genai>=1.73 not installed. Run: pip install 'google-genai>=1.73'"
        )
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) before calling Gemini.")
    return genai.Client()


def _is_pro(model: str) -> bool:
    return "pro" in model.lower()


# ─────────────────────────────────────────────────────────────────────────────
# JSONL builders
# ─────────────────────────────────────────────────────────────────────────────
def _format_context(context: list[str]) -> str:
    return "\n\n".join(f"[문맥 {i + 1}] {c}" for i, c in enumerate(context))


def write_generation_batch_jsonl(
    items: list[dict], model: str, out_path: Path
) -> None:
    """Write a generation batch JSONL file.

    Parameters
    ----------
    items : list of dict
        Each item must have ``qid``, ``question``, ``context`` (list[str]).
    model : str
        Logical model id, recorded in the JSONL ``metadata`` block (the actual
        model is set at ``submit_batch`` time via ``client.batches.create``).
    out_path : Path
        Destination JSONL file.

    Notes
    -----
    * **No output-token cap.** ``generation_config`` is intentionally absent.
    * Gemini batch JSONL format requires a top-level ``key`` and a ``request``
      body that mirrors ``GenerateContentRequest``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            qid = str(it["qid"])
            prompt = RAG_PROMPT.format(
                question=it["question"], context=_format_context(it["context"])
            )
            line = {
                "key": f"gen::{model}::{qid}",
                "request": {
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt}]}
                    ],
                    # No generation_config → no token cap. Temperature default.
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def write_embedding_batch_jsonl(
    texts: list[dict], model: str, out_path: Path
) -> None:
    """Write an embedding batch JSONL file.

    Parameters
    ----------
    texts : list of dict
        Each item must have ``key`` (unique id) and ``text`` (string to embed).
        Optional: ``task_type`` (e.g. ``RETRIEVAL_DOCUMENT``, ``RETRIEVAL_QUERY``,
        ``SEMANTIC_SIMILARITY``). Default: ``RETRIEVAL_DOCUMENT``.
    model : str
        Recorded in the ``key`` prefix for traceability; actual model is set
        at submit time.
    out_path : Path
        Destination JSONL file.

    Notes
    -----
    The SDK as of 1.73 supports ``client.batches.create(...)`` for both
    ``generateContent`` and ``embedContent`` requests; use the embedding shape
    below. If a future SDK exposes a dedicated
    ``client.batches.create_embeddings(...)`` method, swap the submit path in
    ``submit_batch`` — the JSONL format stays the same.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in texts:
            key = str(t["key"])
            task_type = t.get("task_type", "RETRIEVAL_DOCUMENT")
            line = {
                "key": f"emb::{model}::{key}",
                "request": {
                    "content": {
                        "parts": [{"text": t["text"]}],
                        "role": "user",
                    },
                    "task_type": task_type,
                    # No output_dimensionality override — use model default.
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Context-caching for judge
# ─────────────────────────────────────────────────────────────────────────────
def _build_judge_cache_payload(
    question: str, target: str, metric: str, model: str
) -> tuple[str, str]:
    """Return (system_instruction, cached_user_prefix) for a judge cache.

    The cache stores the static prefix so that each of the 39 candidate requests
    only transmits the candidate text. We pad with ``_PAD_SYSTEM`` if the
    resulting payload is below the Pro cache minimum (4096 tok).
    """
    rubric = EVAL_PROMPTS[metric]
    # Render rubric with {generated} still as a placeholder marker — we strip it
    # from the cached prefix so per-request we can append the candidate fresh.
    prefix = rubric.replace(
        "Candidate answer: {generated}\n\n", ""
    ).format(question=question, target=target)

    system_instruction = (
        "You are a strict Korean RAG judge. Output exactly one integer 1-5 and nothing else."
    )

    # Rough token estimate: ~4 chars/token for mixed KR+EN. Pad if small + Pro.
    est_tokens = len(prefix) // 4
    min_tok = CACHE_MIN_TOKENS["pro"] if _is_pro(model) else CACHE_MIN_TOKENS["flash"]
    if est_tokens < min_tok:
        # Prepend the padding system block (counted toward cache size) N times.
        pad = _PAD_SYSTEM
        reps = max(1, (min_tok - est_tokens) * 4 // max(1, len(pad)) + 1)
        prefix = (pad + "\n\n") * reps + prefix

    return system_instruction, prefix


def create_judge_cache(
    question: str, target: str, metric: str, model: str
) -> str:
    """Create a ``CachedContent`` for a (question, metric) cell.

    Returns
    -------
    str
        The cache resource name (e.g. ``cachedContents/abc123``) to feed into
        ``GenerateContentConfig(cached_content=...)``.

    Notes
    -----
    * One cache per (qid, metric) — 300 Q × 4 metrics = 1,200 caches per judge
      run. Billing: 1 write + 38 reads per cell for 39 candidates.
    * TTL is fixed at ``'1h'`` — plenty of headroom for a batch that completes
      within minutes. Extend via ``client.caches.update`` if a job pushes past
      that.
    """
    client = _client()
    system_instruction, cached_user_prefix = _build_judge_cache_payload(
        question, target, metric, model
    )

    cache = client.caches.create(
        model=model,
        config=genai_types.CreateCachedContentConfig(
            display_name=f"judge::{metric}::{hash((question, target)) & 0xFFFFFFFF:08x}",
            system_instruction=system_instruction,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=cached_user_prefix)],
                )
            ],
            ttl="21600s",
        ),
    )
    return cache.name


def write_judge_batch_jsonl_with_cache(
    items: list[dict],
    cache_map: dict[tuple, str],
    judge_model: str,
    out_path: Path,
) -> None:
    """Write a judge batch JSONL that references per-cell context caches.

    Parameters
    ----------
    items : list of dict
        ``[{'custom_id','qid','metric','candidate'}]``.
    cache_map : dict[(qid, metric) -> cache_name]
        Output of repeatedly calling ``create_judge_cache`` for each cell.
    judge_model : str
        Recorded in the JSONL ``key`` prefix; submit with the same model id.
    out_path : Path
        Destination JSONL file.

    Each request body
    -----------------
    * ``cached_content``: the cache name for this (qid, metric).
    * ``contents``: only the candidate text — the rubric/Q/reference come from
      the cache.
    * No ``generation_config`` → **no output-token cap**.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            key = (str(it["qid"]), str(it["metric"]))
            cache_name = cache_map.get(key)
            if not cache_name:
                raise KeyError(f"No cache registered for (qid, metric) = {key}")
            candidate_msg = f"Candidate answer: {it['candidate']}\n\nScore (1-5):"
            line = {
                "key": str(it["custom_id"]),
                "request": {
                    "cached_content": cache_name,
                    "contents": [
                        {"role": "user", "parts": [{"text": candidate_msg}]}
                    ],
                    # Deterministic scoring + thinking 최소 (모델이 thinking 강제).
                    "generation_config": {
                        "temperature": 0.0,
                        "thinking_config": {"thinking_budget": 128},
                    },
                    # NOTE: no max_output_tokens — judge may emit a short prose
                    # trail before the score; parser grabs the last 1-5.
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Batch lifecycle
# ─────────────────────────────────────────────────────────────────────────────
def submit_batch(
    jsonl_path_or_inline: Path | list[dict],
    model: str,
    display_name: str,
    task_type: str = "generate",
) -> str:
    """Submit a batch job and return the job resource name.

    Parameters
    ----------
    jsonl_path_or_inline
        * ``Path``: file is uploaded via ``client.files.upload`` then referenced
          by ``src=<uploaded_file.name>``.
        * ``list[dict]``: passed inline as ``src=<list>`` (bounded to small runs
          per SDK limits — prefer file uploads for >100 requests).
    model : str
        Gemini model id (e.g. ``gemini-3.1-pro``, ``gemini-3.1-flash``,
        ``gemini-embedding-001``).
    display_name : str
        Human-readable job label.
    task_type : str
        ``'generate'`` | ``'embed'`` | ``'judge'`` — currently cosmetic; the
        Gemini batch endpoint routes by model capability. Kept for logging.

    Returns
    -------
    str
        ``batch.name`` — pass to ``poll_batch`` / ``download_batch_output``.
    """
    client = _client()

    if isinstance(jsonl_path_or_inline, (str, Path)):
        path = Path(jsonl_path_or_inline)
        uploaded = client.files.upload(
            file=str(path),
            config={"display_name": f"{display_name}.jsonl", "mime_type": "jsonl"},
        )
        src: Any = uploaded.name
    else:
        # Inline requests — use sparingly; file upload is preferred.
        src = jsonl_path_or_inline

    job = client.batches.create(
        model=model,
        src=src,
        config={"display_name": display_name},
    )
    return job.name


def poll_batch(job_name: str, poll_interval: int = DEFAULT_POLL_INTERVAL) -> dict:
    """Block until the batch hits a terminal state, returning the job object.

    The returned dict has at minimum: ``name``, ``state``, ``dest.file_name``
    (on success), ``error`` (on failure). Uses ``client.batches.get``.
    """
    client = _client()
    while True:
        job = client.batches.get(name=job_name)
        state = getattr(job, "state", None) or getattr(job, "status", None)
        state_str = str(state)
        if state_str in TERMINAL_STATES or any(
            t in state_str for t in TERMINAL_STATES
        ):
            # Serialize to a plain dict for downstream consumers.
            return _job_to_dict(job)
        time.sleep(poll_interval)


def _job_to_dict(job: Any) -> dict:
    """Best-effort conversion of a batch job object to a plain dict."""
    if isinstance(job, dict):
        return job
    out: dict[str, Any] = {}
    for attr in ("name", "state", "status", "display_name", "model", "error"):
        if hasattr(job, attr):
            out[attr] = getattr(job, attr)
    dest = getattr(job, "dest", None)
    if dest is not None:
        out["dest"] = {
            "file_name": getattr(dest, "file_name", None),
            "inlined_responses": getattr(dest, "inlined_responses", None),
        }
    return out


def download_batch_output(job_name: str, out_path: Path) -> dict:
    """Download result JSONL for a finished batch and save to ``out_path``.

    Returns a small stats dict: ``{'lines': int, 'bytes': int, 'path': str}``.
    """
    client = _client()
    job = client.batches.get(name=job_name)
    dest_file_name = getattr(getattr(job, "dest", None), "file_name", None)
    if not dest_file_name:
        raise RuntimeError(
            f"Job {job_name} has no dest.file_name — state={getattr(job, 'state', '?')}"
        )

    blob = client.files.download(file=dest_file_name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # SDK returns bytes for JSONL downloads.
    data = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    out_path.write_bytes(data)
    lines = data.count(b"\n")
    return {"lines": lines, "bytes": len(data), "path": str(out_path)}


# ─────────────────────────────────────────────────────────────────────────────
# Result parsers
# ─────────────────────────────────────────────────────────────────────────────
def _iter_jsonl(path: Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_text(response: dict) -> str:
    """Pull the first text part from a Gemini batch response dict."""
    try:
        cands = response.get("candidates") or []
        for c in cands:
            content = c.get("content") or {}
            for p in content.get("parts") or []:
                if "text" in p and p["text"]:
                    return p["text"]
        # Inlined shape fallback
        if "text" in response:
            return response["text"] or ""
    except Exception:
        pass
    return ""


def _extract_usage(response: dict) -> dict:
    um = response.get("usage_metadata") or response.get("usageMetadata") or {}
    return {
        "prompt_tokens": um.get("prompt_token_count") or um.get("promptTokenCount"),
        "candidates_tokens": um.get("candidates_token_count")
        or um.get("candidatesTokenCount"),
        "cached_tokens": um.get("cached_content_token_count")
        or um.get("cachedContentTokenCount"),
        "total_tokens": um.get("total_token_count") or um.get("totalTokenCount"),
    }


def parse_generation_results(output_jsonl: Path) -> list[dict]:
    """Parse generation batch output into rows.

    Returns
    -------
    list of dict
        Each row: ``{'key','qid','model','text','usage','error'}``.
    """
    rows: list[dict] = []
    for obj in _iter_jsonl(output_jsonl):
        key = obj.get("key", "")
        # key format: gen::<model>::<qid>
        parts = key.split("::", 2)
        model = parts[1] if len(parts) == 3 else None
        qid = parts[2] if len(parts) == 3 else None
        response = obj.get("response") or {}
        err = obj.get("error")
        rows.append(
            {
                "key": key,
                "qid": qid,
                "model": model,
                "text": _extract_text(response) if not err else "",
                "usage": _extract_usage(response) if not err else {},
                "error": err,
            }
        )
    return rows


def parse_embedding_results(output_jsonl: Path) -> list[dict]:
    """Parse embedding batch output into rows.

    Returns
    -------
    list of dict
        Each row: ``{'key','model','text_key','embedding','error'}``.
    """
    rows: list[dict] = []
    for obj in _iter_jsonl(output_jsonl):
        key = obj.get("key", "")
        parts = key.split("::", 2)
        model = parts[1] if len(parts) == 3 else None
        text_key = parts[2] if len(parts) == 3 else None
        response = obj.get("response") or {}
        err = obj.get("error")
        emb = None
        # Response shape: {'embedding': {'values': [...]}} or {'values':[...]}
        emb_obj = response.get("embedding") or response
        values = None
        if isinstance(emb_obj, dict):
            values = emb_obj.get("values")
        if values:
            emb = values
        rows.append(
            {
                "key": key,
                "model": model,
                "text_key": text_key,
                "embedding": emb,
                "error": err,
            }
        )
    return rows


_SCORE_RE = re.compile(r"\b([1-5])\b")


def parse_judge_results(output_jsonl: Path) -> list[dict]:
    """Parse judge batch output → ``[{'custom_id','score','raw','usage'}]``.

    Score extraction mirrors ``scripts/llm_judge.py``: we take the **last**
    ``\\b[1-5]\\b`` match in the raw text. Missing / errored rows get ``0``.
    """
    rows: list[dict] = []
    for obj in _iter_jsonl(output_jsonl):
        custom_id = obj.get("key", "")
        response = obj.get("response") or {}
        err = obj.get("error")
        raw = _extract_text(response) if not err else ""
        score = 0
        if raw:
            matches = _SCORE_RE.findall(raw)
            if matches:
                score = int(matches[-1])
            else:
                # Fallback: any digit 1-5 in reverse
                for ch in reversed(raw):
                    if ch.isdigit() and 1 <= int(ch) <= 5:
                        score = int(ch)
                        break
        rows.append(
            {
                "custom_id": custom_id,
                "score": score,
                "raw": raw,
                "usage": _extract_usage(response),
                "error": err,
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Demo / __main__
# ─────────────────────────────────────────────────────────────────────────────
def _demo(live: bool) -> None:
    """End-to-end shape demo.

    Without ``--live`` this only writes JSONL fixtures to ``/tmp`` and prints
    what *would* be submitted — no network calls, no cache creation.
    """
    tmp = Path("/tmp/gemini_batch_demo")
    tmp.mkdir(exist_ok=True)

    # ── 1. Generation batch ──
    gen_items = [
        {
            "qid": "q001",
            "question": "대한민국 수도는?",
            "context": ["대한민국의 수도는 서울특별시이다.", "면적: 605 km²."],
        }
    ]
    gen_path = tmp / "gen.jsonl"
    write_generation_batch_jsonl(gen_items, model="gemini-3.1-flash", out_path=gen_path)
    print(f"[gen] wrote {gen_path} ({gen_path.stat().st_size}B)")

    # ── 2. Embedding batch ──
    emb_texts = [
        {"key": "doc001", "text": "서울은 대한민국의 수도입니다."},
        {"key": "q001", "text": "대한민국 수도는?", "task_type": "RETRIEVAL_QUERY"},
    ]
    emb_path = tmp / "emb.jsonl"
    write_embedding_batch_jsonl(
        emb_texts, model="gemini-embedding-001", out_path=emb_path
    )
    print(f"[emb] wrote {emb_path} ({emb_path.stat().st_size}B)")

    # ── 3. Judge batch with per-cell caches ──
    judge_model = "gemini-3.1-pro"
    cache_map: dict[tuple, str] = {}
    if live:
        for metric in EVAL_PROMPTS:
            cache_map[("q001", metric)] = create_judge_cache(
                question="대한민국 수도는?",
                target="서울특별시",
                metric=metric,
                model=judge_model,
            )
            print(f"[judge-cache] {metric} → {cache_map[('q001', metric)]}")
    else:
        # Stub cache names for offline shape-check.
        for metric in EVAL_PROMPTS:
            cache_map[("q001", metric)] = f"cachedContents/STUB-q001-{metric}"
        print(f"[judge-cache] (stub) {len(cache_map)} caches registered")

    judge_items = [
        {
            "custom_id": f"q001::{metric}::cand{i}",
            "qid": "q001",
            "metric": metric,
            "candidate": cand,
        }
        for metric in EVAL_PROMPTS
        for i, cand in enumerate(["서울입니다.", "부산입니다.", "잘 모르겠습니다."])
    ]
    judge_path = tmp / "judge.jsonl"
    write_judge_batch_jsonl_with_cache(
        judge_items, cache_map, judge_model=judge_model, out_path=judge_path
    )
    print(
        f"[judge] wrote {judge_path} ({judge_path.stat().st_size}B, "
        f"{len(judge_items)} rows, {len(cache_map)} caches)"
    )

    if not live:
        print("\n(offline mode — skipping submit/poll/parse. Re-run with --live.)")
        return

    # ── 4. Submit → poll → download → parse ──
    job_name = submit_batch(
        gen_path,
        model="gemini-3.1-flash",
        display_name="demo-gen",
        task_type="generate",
    )
    print(f"[submit] job={job_name}")
    job = poll_batch(job_name, poll_interval=30)
    print(f"[poll]   state={job.get('state')}")
    out_path = tmp / "gen_out.jsonl"
    stats = download_batch_output(job_name, out_path)
    print(f"[dl]     {stats}")
    rows = parse_generation_results(out_path)
    print(f"[parse]  {len(rows)} gen rows, first.text={rows[0]['text'][:80]!r}")


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Gemini Batch API provider demo (shape-check by default)."
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Actually hit the Gemini API (needs GOOGLE_API_KEY).",
    )
    args = ap.parse_args()
    try:
        _demo(live=args.live)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
