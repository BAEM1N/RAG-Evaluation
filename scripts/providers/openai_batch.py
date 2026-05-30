#!/usr/bin/env python3
"""OpenAI /v1/batches helpers for the RAG-Evaluation project.

Three task families are supported:
  1. Generation  — Phase 5A candidate LLM answers (chat.completions)
  2. Embedding   — text-embedding-3-small / text-embedding-3-large
  3. Judge       — flagship judge (e.g. gpt-5.4-pro) scoring candidate answers

Design goals
------------
* 50% batch discount + automatic prompt caching on prefixes >= 1024 tokens.
  For the judge task we therefore keep rubric + question + reference as the
  stable prefix of the user message and push the varying "Candidate: ..."
  section to the very end — this maximises OpenAI auto-cache reuse across
  every candidate for a given (question, metric) pair.
* **No output cap.** We never set ``max_tokens``, ``max_completion_tokens``,
  ``max_output_tokens``, or any ``num_predict``-style knob. The model runs to
  natural completion. This rule is non-negotiable (see project spec).
* Auth is taken from ``OPENAI_API_KEY`` (loaded from ``~/.env`` by the shell).
* Unordered results are matched via ``custom_id``.

Rubrics are copied verbatim from ``scripts/llm_judge.py`` so that cloud and
local judges stay methodologically identical to allganize's leaderboard.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    from openai import OpenAI  # openai>=2.31
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise ImportError(
        "openai>=2.31 is required. Install with `pip install 'openai>=2.31'`."
    ) from exc


logger = logging.getLogger("openai_batch")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Rubrics (verbatim port of scripts/llm_judge.py::EVAL_PROMPTS)
# ---------------------------------------------------------------------------

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

# Judge-prompt prefix/suffix: split each rubric at the ``{generated}`` slot so
# that the text before the candidate (rubric + question + reference) is
# byte-identical across every candidate sharing the same (question, metric)
# pair → OpenAI batch auto-cache kicks in once that prefix crosses 1024 tokens.
# The rubric text is preserved exactly (same whitespace, same trailing
# instruction) as in scripts/llm_judge.py::EVAL_PROMPTS.
_JUDGE_SPLIT_PARTS: dict[str, tuple[str, str]] = {
    metric: tuple(prompt.split("{generated}", 1))  # type: ignore[misc]
    for metric, prompt in EVAL_PROMPTS.items()
}
JUDGE_PREFIX_TEMPLATE: dict[str, str] = {m: p[0] for m, p in _JUDGE_SPLIT_PARTS.items()}
JUDGE_SUFFIX_TEMPLATE: dict[str, str] = {m: p[1] for m, p in _JUDGE_SPLIT_PARTS.items()}


RAG_PROMPT = (
    "다음 검색된 문맥을 사용하여 질문에 답하세요. "
    "답을 모르면 모른다고 하세요. 최대 3문장으로 간결하게 답하세요.\n\n"
    "질문: {question}\n\n문맥:\n{context}\n\n답변:"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _client() -> OpenAI:
    """Construct an OpenAI client using OPENAI_API_KEY from the environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it (or ensure ~/.env is loaded)."
        )
    return OpenAI(api_key=api_key)


def _dump_jsonl(records: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _as_context_text(context: Any) -> str:
    """Accept ``list[str]`` or pre-joined string for context."""
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        return "\n\n".join(str(c) for c in context)
    return str(context)


# ---------------------------------------------------------------------------
# JSONL writers
# ---------------------------------------------------------------------------

def write_generation_batch_jsonl(
    items: list[dict],
    model: str,
    out_path: Path,
) -> None:
    """Write a JSONL batch file for Phase 5A candidate generation.

    Each ``items`` element is ``{"qid": str, "question": str, "context": list[str]}``.
    Resulting requests target ``/v1/chat/completions`` with ``temperature=0`` and
    **no output cap** (max_tokens is intentionally omitted).
    """
    records = []
    for it in items:
        qid = it["qid"]
        prompt = RAG_PROMPT.format(
            question=it["question"],
            context=_as_context_text(it.get("context", "")),
        )
        body: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        records.append({
            "custom_id": f"gen::{qid}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })
    _dump_jsonl(records, out_path)
    logger.info("Wrote %d generation requests -> %s", len(records), out_path)


def write_embedding_batch_jsonl(
    texts: list[dict],
    model: str,
    out_path: Path,
) -> None:
    """Write a JSONL batch file for embeddings.

    Each ``texts`` element is ``{"id": str, "text": str}``. Resulting requests
    target ``/v1/embeddings``. No dimensionality override is applied — caller
    may set one upstream if they need shortened ``text-embedding-3-large``
    vectors.
    """
    records = []
    for t in texts:
        tid = t["id"]
        records.append({
            "custom_id": f"emb::{tid}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": model,
                "input": t["text"],
            },
        })
    _dump_jsonl(records, out_path)
    logger.info("Wrote %d embedding requests -> %s", len(records), out_path)


def write_judge_batch_jsonl(
    items: list[dict],
    judge_model: str,
    out_path: Path,
) -> None:
    """Write a JSONL batch file for judge scoring.

    Each ``items`` element must contain ``custom_id``, ``question``, ``target``,
    ``candidate``, and ``metric`` (one of similarity/correctness/completeness/
    faithfulness).

    The user-message is assembled in two blocks so that the first block
    (rubric + question + reference) stays byte-identical across all candidates
    sharing the same (question, metric) pair. OpenAI's batch auto-cache kicks
    in once that prefix crosses 1024 tokens, yielding ~75% discount on the
    cached portion. The only varying part is the candidate text and the
    trailing "Respond with ONLY the integer 1-5." instruction, which the
    rubric already contains — we keep both ends intact for methodological
    parity with ``llm_judge.py``.
    """
    if items and "metric" not in items[0]:
        raise ValueError("write_judge_batch_jsonl: items[*] must include 'metric'")

    records = []
    for it in items:
        metric = it["metric"]
        if metric not in EVAL_PROMPTS:
            raise ValueError(f"unknown metric: {metric!r}")

        prefix = JUDGE_PREFIX_TEMPLATE[metric].format(
            question=it["question"],
            target=it["target"],
        )
        suffix = JUDGE_SUFFIX_TEMPLATE[metric]
        user_content = f"{prefix}{it['candidate']}{suffix}"

        body: dict[str, Any] = {
            "model": judge_model,
            "temperature": 0,
            "messages": [{"role": "user", "content": user_content}],
        }
        records.append({
            "custom_id": it["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })
    _dump_jsonl(records, out_path)
    logger.info("Wrote %d judge requests -> %s", len(records), out_path)


# ---------------------------------------------------------------------------
# Submit / poll / download
# ---------------------------------------------------------------------------

_VALID_ENDPOINTS = {"/v1/chat/completions", "/v1/embeddings"}


def submit_batch(jsonl_path: Path, endpoint: str) -> str:
    """Upload the JSONL file and create a batch job. Returns ``batch_id``."""
    if endpoint not in _VALID_ENDPOINTS:
        raise ValueError(f"endpoint must be one of {_VALID_ENDPOINTS}, got {endpoint!r}")
    client = _client()

    logger.info("Uploading %s (purpose=batch)", jsonl_path)
    with Path(jsonl_path).open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    logger.info("  file_id=%s", file_obj.id)

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        completion_window="24h",
        metadata={"source": "RAG-Evaluation", "file": Path(jsonl_path).name},
    )
    logger.info("  batch_id=%s status=%s", batch.id, batch.status)
    return batch.id


_TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}


def poll_batch(batch_id: str, poll_interval: int = 60) -> dict:
    """Block until the batch reaches a terminal state; return the Batch object dict."""
    client = _client()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = getattr(batch, "request_counts", None)
        counts_str = ""
        if counts is not None:
            counts_str = (
                f" (total={getattr(counts, 'total', '?')} "
                f"completed={getattr(counts, 'completed', '?')} "
                f"failed={getattr(counts, 'failed', '?')})"
            )
        logger.info("batch %s status=%s%s", batch_id, status, counts_str)

        if status in _TERMINAL_STATES:
            return batch.model_dump() if hasattr(batch, "model_dump") else dict(batch)

        time.sleep(poll_interval)


def download_batch_output(batch_id: str, out_dir: Path) -> dict:
    """Download the output (and error, if any) file for a completed batch.

    Returns ``{'output': Path|None, 'error': Path|None, 'stats': {...}}``.
    """
    client = _client()
    batch = client.batches.retrieve(batch_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"output": None, "error": None, "stats": {}}
    output_id = getattr(batch, "output_file_id", None)
    error_id = getattr(batch, "error_file_id", None)

    if output_id:
        dst = out_dir / f"{batch_id}__output.jsonl"
        content = client.files.content(output_id).read()
        dst.write_bytes(content)
        result["output"] = dst
        logger.info("Downloaded output file -> %s (%d bytes)", dst, len(content))

    if error_id:
        dst = out_dir / f"{batch_id}__error.jsonl"
        content = client.files.content(error_id).read()
        dst.write_bytes(content)
        result["error"] = dst
        logger.info("Downloaded error file -> %s (%d bytes)", dst, len(content))

    counts = getattr(batch, "request_counts", None)
    if counts is not None:
        result["stats"] = {
            "total": getattr(counts, "total", None),
            "completed": getattr(counts, "completed", None),
            "failed": getattr(counts, "failed", None),
            "status": batch.status,
        }
    else:
        result["stats"] = {"status": batch.status}
    return result


# ---------------------------------------------------------------------------
# Result parsers
# ---------------------------------------------------------------------------

def _extract_usage(resp_body: dict) -> dict:
    """Pull the usage block (chat or embedding) from a per-line response body."""
    usage = resp_body.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _extract_cached_tokens(resp_body: dict) -> int:
    """Return ``prompt_tokens_details.cached_tokens`` if present, else 0."""
    usage = resp_body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens", 0) or 0)


def parse_generation_results(output_jsonl: Path) -> list[dict]:
    """Parse generation batch output into per-``custom_id`` records."""
    out: list[dict] = []
    for line in _iter_jsonl(Path(output_jsonl)):
        custom_id = line.get("custom_id")
        err = line.get("error")
        resp = line.get("response") or {}
        body = resp.get("body") or {}

        if err or not body:
            out.append({
                "custom_id": custom_id,
                "content": "",
                "usage": {},
                "cached_tokens": 0,
                "error": err or "empty response body",
            })
            continue

        choices = body.get("choices") or []
        content = ""
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""

        out.append({
            "custom_id": custom_id,
            "content": content,
            "usage": _extract_usage(body),
            "cached_tokens": _extract_cached_tokens(body),
        })
    return out


def parse_embedding_results(output_jsonl: Path) -> list[dict]:
    """Parse embedding batch output into per-``custom_id`` records."""
    out: list[dict] = []
    for line in _iter_jsonl(Path(output_jsonl)):
        custom_id = line.get("custom_id")
        err = line.get("error")
        resp = line.get("response") or {}
        body = resp.get("body") or {}

        if err or not body:
            out.append({
                "custom_id": custom_id,
                "embedding": [],
                "usage": {},
                "error": err or "empty response body",
            })
            continue

        data = body.get("data") or []
        embedding: list[float] = []
        if data:
            embedding = data[0].get("embedding") or []

        out.append({
            "custom_id": custom_id,
            "embedding": embedding,
            "usage": _extract_usage(body),
        })
    return out


_DIGIT_RE = re.compile(r"\b([1-5])\b")
_ANY_DIGIT_RE = re.compile(r"([1-5])")


def _score_from_text(s: str) -> int:
    """Extract 1-5 integer score from model output, 0 on failure.

    Strategy (matches scripts/llm_judge.py):
      1) last regex match of \\b[1-5]\\b
      2) any digit 1-5 anywhere (iterate from end)
      3) otherwise 0
    """
    if not s:
        return 0
    matches = _DIGIT_RE.findall(s)
    if matches:
        return int(matches[-1])
    for c in reversed(s):
        if c.isdigit():
            n = int(c)
            if 1 <= n <= 5:
                return n
    return 0


def parse_judge_results(output_jsonl: Path) -> list[dict]:
    """Parse judge batch output; each record carries an integer score 0-5."""
    out: list[dict] = []
    for line in _iter_jsonl(Path(output_jsonl)):
        custom_id = line.get("custom_id")
        err = line.get("error")
        resp = line.get("response") or {}
        body = resp.get("body") or {}

        if err or not body:
            out.append({
                "custom_id": custom_id,
                "score": 0,
                "raw_content": "",
                "usage": {},
                "cached_tokens": 0,
                "error": err or "empty response body",
            })
            continue

        choices = body.get("choices") or []
        content = ""
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""

        score = _score_from_text(content)
        out.append({
            "custom_id": custom_id,
            "score": score,
            "raw_content": content,
            "usage": _extract_usage(body),
            "cached_tokens": _extract_cached_tokens(body),
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    parser = argparse.ArgumentParser(description="OpenAI batch driver for RAG-Evaluation")
    parser.add_argument(
        "--task",
        choices=["generation", "embedding", "judge"],
        required=True,
        help="Task family; determines the endpoint used when --submit.",
    )
    parser.add_argument("--submit", type=Path, help="Path to JSONL batch file to submit.")
    parser.add_argument("--poll", type=str, help="Batch ID to poll until terminal.")
    parser.add_argument(
        "--download",
        type=str,
        help="Batch ID to download output/error files for.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/openai_batch"),
        help="Directory to write downloaded output files into.",
    )
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    actions = [a for a in (args.submit, args.poll, args.download) if a]
    if len(actions) != 1:
        parser.error("provide exactly one of --submit / --poll / --download")

    endpoint = "/v1/embeddings" if args.task == "embedding" else "/v1/chat/completions"

    if args.submit:
        batch_id = submit_batch(args.submit, endpoint=endpoint)
        print(batch_id)
        return 0

    if args.poll:
        result = poll_batch(args.poll, poll_interval=args.poll_interval)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.download:
        res = download_batch_output(args.download, args.out_dir)
        print(json.dumps({
            "output": str(res["output"]) if res["output"] else None,
            "error": str(res["error"]) if res["error"] else None,
            "stats": res["stats"],
        }, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(_cli())
