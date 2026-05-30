#!/usr/bin/env python3
"""Anthropic Message Batches provider for RAG-Evaluation Phase 5A.

Two tasks are supported:
  1. generation — candidate LLM answers against the Anthropic direct API
     (claude-opus-4-7 / claude-sonnet-4-6 / claude-sonnet-4-5 / claude-haiku-4-5)
  2. judge      — flagship judge (claude-opus-4-7) scoring candidates on 4
     metrics (similarity, correctness, completeness, faithfulness), 1-5 scale,
     rubrics ported verbatim from scripts/llm_judge.py EVAL_PROMPTS

Embeddings are NOT supported by Anthropic — no embedding task here.

Key design points
-----------------
* Explicit prompt caching with ``cache_control={"type":"ephemeral","ttl":"1h"}``
  on the (Q, metric)-invariant prefix. Anthropic cache reads are billed at
  0.1× base input (90% off) — the biggest discount among providers. For the
  judge task this matters: the rubric + (Q, target) prefix is written once per
  (Q, metric) pair and then read 38× across the 39 candidates.

* Minimum cacheable prefix size is model-dependent — 4096 tokens for Opus 4.7
  and Haiku 4.5, 2048 tokens for Sonnet 4.6. Shorter prefixes are silently not
  cached (no error; ``cache_creation_input_tokens`` just stays 0). Callers
  decide breakpoints only where the block is large enough.

* ``max_tokens`` is a required Anthropic parameter (unlike OpenAI). A hard low
  cap is explicitly forbidden by the project. We default to 8192 as a
  "no practical limit" value; callers can pass higher if a model supports it.

* Opus 4.7 specifics: sampling parameters (``temperature`` / ``top_p`` /
  ``top_k``) are removed — sending any of them returns 400. ``budget_tokens``
  is also removed. Adaptive thinking is off by default; we leave it off for
  judge calls (deterministic short integer) and off for generation (caller can
  extend later).

* Batch endpoint: ``client.messages.batches.create(requests=[Request(...)])``.
  Limits: 100,000 requests or 256 MB per batch; results retained 29 days.
  Polling: ``client.messages.batches.retrieve(batch_id)`` until
  ``processing_status == "ended"``; results via
  ``client.messages.batches.results(batch_id)`` (JSONL stream).

Env
---
  ANTHROPIC_API_KEY — required. Auth is via the official anthropic SDK
  (``>=0.97``; declared in pyproject.toml).
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
from typing import Iterable

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("anthropic-batch")


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────

GEN_MODELS = (
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
)
JUDGE_MODEL_DEFAULT = "claude-opus-4-7"

# Minimum cacheable prefix in tokens. Below this, cache_control is a no-op
# (silently — cache_creation_input_tokens stays 0).
# Source: shared/prompt-caching.md (skill docs).
MIN_CACHEABLE_TOKENS = {
    "claude-opus-4-7":   4096,
    "claude-opus-4-6":   4096,
    "claude-opus-4-5":   4096,
    "claude-haiku-4-5":  4096,
    "claude-sonnet-4-6": 2048,
    "claude-sonnet-4-5": 1024,
}

# "No practical limit" default. Anthropic requires max_tokens, but the project
# forbids tight caps. 8192 is a safe floor for short judge integers AND long
# generation answers; callers can raise this for long-form generation.
NO_CAP_MAX_TOKENS = 8192

# Explicit prompt-cache breakpoint — 1h TTL is the right choice for batch work
# because a single batch can take up to 1h to complete. The 2× write cost of
# the 1h TTL pays off quickly when the prefix is reused across 38+ candidates.
CACHE_EPHEMERAL_1H = {"type": "ephemeral", "ttl": "1h"}


# ──────────────────────────────────────────────────────────────────────────
# Judge rubrics — copied verbatim from scripts/llm_judge.py EVAL_PROMPTS.
# Do NOT edit without also editing the reference in llm_judge.py; the whole
# point of the judge task is that the rubric is identical across providers.
# ──────────────────────────────────────────────────────────────────────────

EVAL_PROMPTS = {
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


# Generation system / RAG prompt — mirrors scripts/phase5_provider.RAG_PROMPT.
# The system text is the same across every question and so is an ideal cache
# target when (and only when) the accompanying context block pushes the full
# prefix above the per-model minimum.
GEN_SYSTEM_TEXT = (
    "다음 검색된 문맥을 사용하여 질문에 답하세요. "
    "답을 모르면 모른다고 하세요. 최대 3문장으로 간결하게 답하세요."
)


# ──────────────────────────────────────────────────────────────────────────
# Judge prefix padding — mirrors gemini_batch._PAD_SYSTEM.
# A neutral Korean-RAG judging guidance block, distinct from the 4 rubrics in
# EVAL_PROMPTS. Used by write_judge_batch_requests to inflate the cached
# prefix above MIN_CACHEABLE_TOKENS (4096 for Opus 4.7 / Haiku 4.5, 2048 for
# Sonnet 4.6). One repetition is ~300-500 tokens of mixed KR/EN — the builder
# repeats it as many times as needed to clear the per-model threshold.
# ──────────────────────────────────────────────────────────────────────────

_PAD_SYSTEM = (
    "You are a meticulous bilingual (Korean/English) judge for Korean "
    "retrieval-augmented generation. Apply the metric rubric strictly and "
    "literally — do not invent new criteria and do not soften the scale. "
    "Korean surface variants (존댓말/반말, 조사 차이, 한자어 ↔ 고유어 치환, "
    "띄어쓰기 변형)은 의미가 동일하면 동등하게 취급하되, 숫자·날짜·단위·고유명사는 "
    "참조와 정확히 일치해야 한다. 후보 답변이 참조에 없는 내용을 추가했다면 "
    "faithfulness 와 completeness 축에서 그 정도에 비례해 감점한다. 반대로 "
    "참조에는 있는 핵심 포인트를 누락한 경우 completeness 와 correctness 축에서 "
    "감점한다. 후보가 '모릅니다' / refusal / empty 로 응답했고 참조에 내용이 "
    "존재한다면 correctness 와 completeness 는 1 로 고정한다. 부분 일치는 "
    "1-5 척도에서 정의된 중간 등급 (2 = major divergence, 3 = moderate, "
    "4 = mostly aligned)을 문자 그대로 적용하며, 문체·어조·서식 차이는 의미가 "
    "보존되는 한 감점 사유가 아니다. 출력은 반드시 1~5 사이 정수 한 글자만 내보내고 "
    "설명, 마크다운, JSON, 따옴표, 접두어/접미어를 포함하지 않는다. 이 지침은 "
    "동일 (question, metric) 셀에 속한 39개 후보 전부에 재사용되는 캐시된 "
    "프리픽스이므로, 프리픽스를 훼손하지 않도록 후보 텍스트를 system 블록이 "
    "아닌 user 메시지에만 넣는다. 판정은 결정론적이어야 하며, 같은 입력에 대해 "
    "두 번 채점할 경우 동일 점수를 반환해야 한다."
)


def _estimate_tokens(text: str) -> int:
    """Cheap local token estimator for Korean-heavy mixed text.

    Uses ``len(text) / 3.3`` which is a well-known rule-of-thumb for Korean
    (CJK characters encode to ~1 token each under BPE/tiktoken-like tokenizers,
    ASCII runs at ~4 chars/token). Good enough to decide how many pad repeats
    are needed — we do NOT add a tiktoken dependency just for this.
    """
    if not text:
        return 0
    return int(len(text) / 3.3) + 1


def _build_padded_system_blocks(
    rubric_only: str,
    qref_block: str,
    min_tokens: int,
    headroom: int = 500,
) -> list[dict]:
    """Build the 3-block system array for a judge request with a pad block
    sized so the total prefix clears ``min_tokens + headroom`` on this model.

    Returns three content blocks:
      (a) rubric_only              — stable across the entire (Q, metric, cand)
                                     fan-out; cache_control breakpoint 1.
      (b) pad block                — _PAD_SYSTEM repeated enough times to push
                                     the prefix past the per-model minimum.
                                     NOT individually marked (the breakpoint
                                     on (c) captures it as part of the cached
                                     prefix).
      (c) Question / Reference     — stable across the 39 candidates for a
                                     given (Q, metric); cache_control
                                     breakpoint 2.
    """
    rubric_tokens = _estimate_tokens(rubric_only)
    qref_tokens = _estimate_tokens(qref_block)
    pad_unit_tokens = _estimate_tokens(_PAD_SYSTEM)

    target = min_tokens + headroom
    baseline = rubric_tokens + qref_tokens
    needed = max(0, target - baseline)
    # +1 repetition to bias above the threshold rather than grazing it.
    reps = (needed // max(1, pad_unit_tokens)) + 1 if needed > 0 else 1
    pad_text = (_PAD_SYSTEM + "\n\n") * reps

    return [
        {
            "type": "text",
            "text": rubric_only,
            "cache_control": CACHE_EPHEMERAL_1H,
        },
        {
            "type": "text",
            "text": pad_text,
        },
        {
            "type": "text",
            "text": qref_block,
            "cache_control": CACHE_EPHEMERAL_1H,
        },
    ]


# ──────────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    """Construct an SDK client. Errors if ANTHROPIC_API_KEY is absent so the
    caller gets a clear failure instead of a silent 401 later."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — required for anthropic_batch."
        )
    return anthropic.Anthropic()


# ──────────────────────────────────────────────────────────────────────────
# Request builders
# ──────────────────────────────────────────────────────────────────────────

def write_generation_batch_requests(
    items: list[dict],
    model: str,
    max_tokens: int = NO_CAP_MAX_TOKENS,
) -> list[Request]:
    """Build batch requests for Phase 5A candidate generation.

    ``items`` schema:
        [{"qid": str, "question": str, "context": list[str]}]

    Each request is shaped so the system block (prompt + context) is the
    stable prefix and the user turn carries only the volatile question. The
    ``cache_control`` breakpoint sits on the last system block; Anthropic's
    render order is ``tools → system → messages``, so placing the marker on
    system caches everything before the user turn.

    Caveat — cache hit rate for generation is useful ONLY when the same
    (question, context) pair is queried by multiple models in sequence, or
    when you deliberately dedupe across a batch. For single-pass generation
    per (Q, model) the cache is written once and never read; the 1.25× write
    premium still applies (or 2× for the 1h TTL). Callers who know their
    access pattern is single-pass can drop ``cache_control`` from the returned
    requests to avoid the write premium.

    We still emit the breakpoint by default because:
      * the larger win is on the judge task, which reuses this helper's shape
      * Phase 5A does dedupe by (Q, context) across candidate models
      * below MIN_CACHEABLE_TOKENS the marker is a silent no-op anyway
    """
    model_min = MIN_CACHEABLE_TOKENS.get(model, 4096)
    logger.info(
        "generation: model=%s, items=%d, cache_min_tokens=%d (below this the "
        "cache_control marker is a silent no-op)",
        model, len(items), model_min,
    )

    requests: list[Request] = []
    for it in items:
        qid = it["qid"]
        question = it["question"]
        context_blocks = it["context"]
        context_text = "\n\n".join(context_blocks) if isinstance(context_blocks, list) else str(context_blocks)

        system_blocks = [
            # Fixed instruction — tiny, but first in the prefix so it stays
            # cache-stable across every (Q, context).
            {"type": "text", "text": GEN_SYSTEM_TEXT},
            # Retrieved context — the volume driver. When this block plus the
            # fixed instruction exceeds model_min tokens, cache_control takes
            # effect and the entire system prefix caches for 1h.
            {
                "type": "text",
                "text": f"문맥:\n{context_text}",
                "cache_control": CACHE_EPHEMERAL_1H,
            },
        ]

        params = MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": f"질문: {question}"}],
        )
        requests.append(Request(custom_id=str(qid), params=params))
    return requests


def write_judge_batch_requests(
    items: list[dict],
    judge_model: str = JUDGE_MODEL_DEFAULT,
    max_tokens: int = NO_CAP_MAX_TOKENS,
) -> list[Request]:
    """Build batch requests for flagship judge evaluation.

    ``items`` schema:
        [{"custom_id": str, "question": str, "target": str, "candidate": str,
          "metric": one of EVAL_PROMPTS keys}]

    Prompt structure is optimized for prefix-cache reuse across the 39 ×
    ``metric`` fan-out:

      system = [
        {"type":"text", "text": rubric_for_metric,
         "cache_control": {"type":"ephemeral","ttl":"1h"}},   # (a)
        {"type":"text", "text": _PAD_SYSTEM * N},             # (b)
        {"type":"text", "text": f"Question: ...\\nReference: ...",
         "cache_control": {"type":"ephemeral","ttl":"1h"}},   # (c)
      ]
      messages = [{"role":"user", "content":
                   f"Candidate answer: ...\\nRespond with ONLY the integer 1-5."}]

    Block (a) — the metric rubric — is stable across ALL 300 × 4 = 1,200
    (Q, metric) cells. Block (c) — the Question+Reference pair — is stable
    across the 39 candidates for a given (Q, metric). Block (b) is neutral
    judging guidance (``_PAD_SYSTEM`` repeated) whose sole job is to push the
    cached prefix past ``MIN_CACHEABLE_TOKENS[judge_model]``; it is not
    individually marked because the cache_control marker on (c) captures
    everything from the start of system up to (c) as the cached prefix.

    Padding is computed per-request using a cheap ``len(text)/3.3`` estimator
    (see ``_estimate_tokens``) so we do NOT take a new dependency on tiktoken.
    The estimator targets ``min_tokens + 500`` of headroom so a slightly
    different token boundary in the real tokenizer still clears the threshold.

    No ``max_tokens`` tightening — we pass through ``NO_CAP_MAX_TOKENS``
    (8192) so the judge is never truncated on the output side.
    """
    min_tokens = MIN_CACHEABLE_TOKENS.get(judge_model, 4096)
    logger.info(
        "judge: model=%s, items=%d, cache_min_tokens=%d — system prefix will "
        "be padded with _PAD_SYSTEM to clear this threshold (+500 headroom).",
        judge_model, len(items), min_tokens,
    )

    requests: list[Request] = []
    logged_prefix_estimate = False
    for it in items:
        custom_id = str(it["custom_id"])
        question = it["question"]
        target = it["target"]
        candidate = it["candidate"]
        metric = it["metric"]

        if metric not in EVAL_PROMPTS:
            raise ValueError(
                f"unknown metric {metric!r}; expected one of {list(EVAL_PROMPTS)}"
            )

        # Render the rubric but with the candidate slot stripped out. The
        # rubric text ends at "Respond with ONLY the integer ..." which is
        # moved to the user turn.
        #
        # EVAL_PROMPTS[metric] is a template with {question}, {target},
        # {generated} — we fill {question} and {target} into the cached
        # prefix, and put {generated} into the volatile user turn.
        template = EVAL_PROMPTS[metric]

        # Rubric-only (no question / target / candidate filled in) — stable
        # across every (Q, metric, candidate) triple, so it caches indefinitely
        # within a 1h window.
        header_end = template.index("Question:")
        rubric_only = template[:header_end].rstrip()

        # Rubric body — everything between 'Question:' and 'Respond with':
        rubric_tail_start = template.rindex("Respond with")
        rubric_scale = template[template.index("Rate"):rubric_tail_start].rstrip()

        # The (Q, target) block varies per question but is stable across the
        # 39 candidates evaluating that question.
        qref_block = (
            f"Question: {question}\n"
            f"Reference (ground truth): {target}\n\n"
            f"{rubric_scale}"
        )

        system_blocks = _build_padded_system_blocks(
            rubric_only=rubric_only,
            qref_block=qref_block,
            min_tokens=min_tokens,
            headroom=500,
        )

        if not logged_prefix_estimate:
            est = sum(_estimate_tokens(b["text"]) for b in system_blocks)
            logger.info(
                "judge: estimated padded system prefix ≈ %d tokens "
                "(target ≥ %d = min %d + 500 headroom)",
                est, min_tokens + 500, min_tokens,
            )
            logged_prefix_estimate = True

        messages = [{
            "role": "user",
            "content": (
                f"Candidate answer: {candidate}\n\n"
                "Respond with ONLY the integer 1, 2, 3, 4, or 5."
            ),
        }]

        params = MessageCreateParamsNonStreaming(
            model=judge_model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=messages,
        )
        requests.append(Request(custom_id=custom_id, params=params))
    return requests


# ──────────────────────────────────────────────────────────────────────────
# Submit / poll / collect
# ──────────────────────────────────────────────────────────────────────────

def submit_batch(
    requests: list[Request],
    save_path: Path | None = None,
    chunk_size: int = 5000,
) -> list[str]:
    """Create batches in chunks of ``chunk_size`` requests; return list of batch ids.

    Anthropic batches API has a 256 MB request-body limit; with our padded
    rubric-prefix (~25 KB/request) a single 30K-request batch hits the cloudflare
    413 wall. Splitting into 5K chunks keeps each upload well under the limit.

    If ``save_path`` is provided, write all submitted requests to disk as JSONL
    for reproducibility — useful since the Anthropic batches API does NOT let
    you re-download the original request bodies (only the results).
    """
    if not requests:
        raise ValueError("requests is empty; nothing to submit")

    client = _client()
    batch_ids: list[str] = []
    n = len(requests)
    for i in range(0, n, chunk_size):
        chunk = requests[i : i + chunk_size]
        logger.info(
            "submitting batch chunk %d-%d (%d requests, total=%d)",
            i, i + len(chunk), len(chunk), n,
        )
        batch = client.messages.batches.create(requests=chunk)
        logger.info("  batch created: id=%s, status=%s", batch.id, batch.processing_status)
        batch_ids.append(batch.id)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req, ensure_ascii=False, default=_json_default))
                f.write("\n")
        logger.info("submitted requests saved: %s", save_path)

    return batch_ids


def poll_batch(batch_id: str, poll_interval: int = 60) -> dict:
    """Block until ``processing_status == 'ended'``. Return the final batch
    object as a dict.

    Anthropic batches usually finish within an hour; the 1h cache TTL is
    chosen to survive that worst case. We poll every ``poll_interval`` s
    (default 60) — don't hammer the API.
    """
    client = _client()
    last_status = None
    t_start = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        if status != last_status:
            logger.info(
                "batch %s: status=%s, processing=%s, succeeded=%s, errored=%s",
                batch_id, status,
                getattr(counts, "processing", "?"),
                getattr(counts, "succeeded", "?"),
                getattr(counts, "errored", "?"),
            )
            last_status = status
        if status == "ended":
            elapsed = time.time() - t_start
            logger.info("batch %s ended after %.0fs", batch_id, elapsed)
            return batch.model_dump() if hasattr(batch, "model_dump") else dict(batch)
        time.sleep(poll_interval)


def collect_batch_results(batch_id: str, out_path: Path) -> dict:
    """Download all batch results and write them to ``out_path`` (JSONL).

    Returns a summary dict:
        {
          'total': int, 'succeeded': int, 'errored': int,
          'cache_creation_tokens': int, 'cache_read_tokens': int,
          'output_tokens': int,
        }

    Cache-token accounting is reported back so the caller can verify that
    prompt caching is actually firing. If ``cache_read_tokens == 0`` across a
    batch with many supposedly-cacheable requests, the prefix never met the
    per-model minimum (see ``MIN_CACHEABLE_TOKENS``) or a silent invalidator
    flipped a byte in the prefix — see shared/prompt-caching.md audit table.
    """
    client = _client()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = succeeded = errored = 0
    cache_creation = cache_read = output_tokens = 0

    with out_path.open("w", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch_id):
            total += 1
            # Serialize each result as one JSON line.
            dumped = (
                result.model_dump() if hasattr(result, "model_dump") else dict(result)
            )
            f.write(json.dumps(dumped, ensure_ascii=False, default=_json_default))
            f.write("\n")

            rtype = result.result.type
            if rtype == "succeeded":
                succeeded += 1
                msg = result.result.message
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    cache_creation += getattr(usage, "cache_creation_input_tokens", 0) or 0
                    cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
                    output_tokens += getattr(usage, "output_tokens", 0) or 0
            else:
                errored += 1

    summary = {
        "total": total,
        "succeeded": succeeded,
        "errored": errored,
        "cache_creation_tokens": cache_creation,
        "cache_read_tokens": cache_read,
        "output_tokens": output_tokens,
    }
    logger.info("collected %s → %s: %s", batch_id, out_path, summary)
    return summary


# ──────────────────────────────────────────────────────────────────────────
# Result parsing
# ──────────────────────────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_text_and_usage(msg: dict) -> tuple[str, dict]:
    """Pull concatenated text + usage from an Anthropic Message dict."""
    content_blocks = msg.get("content", []) or []
    parts = []
    for b in content_blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    text = "".join(parts)
    usage = msg.get("usage", {}) or {}
    return text, usage


def parse_generation_results(results_path: Path) -> list[dict]:
    """Parse a JSONL file produced by ``collect_batch_results`` for the
    generation task. Returns ``[{'custom_id','content','usage'}, ...]``.
    Errored rows get ``content=''`` and an ``error`` key."""
    out: list[dict] = []
    for row in _iter_jsonl(results_path):
        custom_id = row.get("custom_id")
        result = row.get("result", {}) or {}
        if result.get("type") == "succeeded":
            content, usage = _extract_text_and_usage(result.get("message", {}) or {})
            out.append({
                "custom_id": custom_id,
                "content": content,
                "usage": usage,
            })
        else:
            err = result.get("error") or {}
            out.append({
                "custom_id": custom_id,
                "content": "",
                "usage": {},
                "error": err.get("type") or result.get("type"),
                "error_message": (err.get("message") if isinstance(err, dict) else None),
            })
    return out


def _last_score(s: str) -> int:
    """Return the last 1-5 integer in ``s``, else 0.

    Mirrors the 3-level fallback parser used in scripts/llm_judge.py.
    """
    if not s:
        return 0
    m = re.findall(r"\b([1-5])\b", s)
    if m:
        return int(m[-1])
    for c in reversed(s):
        if c.isdigit():
            n = int(c)
            if 1 <= n <= 5:
                return n
    return 0


def parse_judge_results(results_path: Path) -> list[dict]:
    """Parse a judge-task result JSONL.

    Returns ``[{'custom_id','score','raw_content','usage'}, ...]``, where
    ``score`` is 1-5 or 0 when no valid integer could be recovered.

    Fallback extraction — same 3-level chain as ``scripts/llm_judge.py``:
      1. last ``\\b[1-5]\\b`` in the content text
      2. after stripping any ``<think>...</think>`` block (Anthropic models
         don't wrap reasoning in those tags, but we keep the fallback
         identical to the in-house judge to avoid provider-specific quirks)
      3. cache diagnostic — compare aggregate ``cache_read_input_tokens`` to
         aggregate total input tokens. If the cache-read ratio is below 0.3
         across the whole batch, log a WARNING that prompt caching likely
         failed and report the observed prefix-token estimate (mean of
         ``cache_creation_input_tokens + cache_read_input_tokens +
         input_tokens`` per request). Ratios close to 1.0 are the happy
         path — every request after the first is served from cache.
    """
    out: list[dict] = []

    # Aggregate token counters for the cache-read ratio diagnostic.
    total_input = 0          # all input tokens charged (incl. cached/uncached)
    total_cache_read = 0     # tokens served from the cache at 0.1× price
    total_cache_write = 0    # tokens that wrote the cache (this run)
    prefix_size_samples: list[int] = []

    for row in _iter_jsonl(results_path):
        custom_id = row.get("custom_id")
        result = row.get("result", {}) or {}
        if result.get("type") != "succeeded":
            err = result.get("error") or {}
            out.append({
                "custom_id": custom_id,
                "score": 0,
                "raw_content": "",
                "usage": {},
                "error": err.get("type") or result.get("type"),
                "error_message": (err.get("message") if isinstance(err, dict) else None),
            })
            continue

        content, usage = _extract_text_and_usage(result.get("message", {}) or {})

        # 1) direct extraction
        score = _last_score(content)
        if score == 0:
            # 2) <think>-stripped fallback (see llm_judge.py)
            stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            score = _last_score(stripped)

        # 3) cache-read aggregation — per-request the observed prefix size
        # is (input_tokens_non_cached + cache_creation + cache_read).
        input_tokens = int(usage.get("input_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
        total_input += input_tokens + cache_read + cache_creation
        total_cache_read += cache_read
        total_cache_write += cache_creation
        prefix_size_samples.append(input_tokens + cache_read + cache_creation)

        out.append({
            "custom_id": custom_id,
            "score": score,
            "raw_content": content,
            "usage": usage,
        })

    # Cache-read ratio diagnostic.
    if total_input > 0:
        ratio = total_cache_read / total_input
        mean_prefix = (
            sum(prefix_size_samples) / len(prefix_size_samples)
            if prefix_size_samples else 0
        )
        if ratio < 0.3:
            logger.warning(
                "judge parse: cache_read / total_input = %.3f (< 0.30) across "
                "%d succeeded rows — prompt caching likely FAILED. "
                "Observed mean prefix ≈ %.0f tokens; cache_read=%d, "
                "cache_write=%d, total_input=%d. If mean prefix is below the "
                "per-model minimum (Opus/Haiku 4096, Sonnet 4.6 2048), raise "
                "_PAD_SYSTEM repetition count in _build_padded_system_blocks.",
                ratio, len(prefix_size_samples), mean_prefix,
                total_cache_read, total_cache_write, total_input,
            )
        else:
            logger.info(
                "judge parse: cache_read / total_input = %.3f across %d rows; "
                "mean observed prefix ≈ %.0f tokens (cache_read=%d, "
                "cache_write=%d, total_input=%d).",
                ratio, len(prefix_size_samples), mean_prefix,
                total_cache_read, total_cache_write, total_input,
            )

    return out


# ──────────────────────────────────────────────────────────────────────────
# Misc helpers
# ──────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    """Best-effort JSON fallback for SDK objects that aren't plain dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return str(obj)


# ──────────────────────────────────────────────────────────────────────────
# __main__ — local end-to-end sanity check (request building ONLY, no HTTP).
# The user forbade actually submitting real batches. This entrypoint just
# builds a small set of fake requests, prints them, and stops.
# ──────────────────────────────────────────────────────────────────────────

def _main() -> None:
    p = argparse.ArgumentParser(description="anthropic_batch request-building smoke test")
    p.add_argument(
        "--task", choices=("generation", "judge"), default="judge",
        help="Which request shape to build for inspection.",
    )
    p.add_argument("--model", default=None, help="Override model id.")
    p.add_argument("--n", type=int, default=2, help="Fake item count.")
    args = p.parse_args()

    if args.task == "generation":
        model = args.model or "claude-opus-4-7"
        items = [
            {
                "qid": f"q{i}",
                "question": f"테스트 질문 {i}번입니다.",
                "context": [f"문맥 조각 A-{i}", f"문맥 조각 B-{i}"],
            }
            for i in range(args.n)
        ]
        reqs = write_generation_batch_requests(items, model=model)
    else:
        model = args.model or JUDGE_MODEL_DEFAULT
        metrics = list(EVAL_PROMPTS.keys())
        items = [
            {
                "custom_id": f"judge_q{i}_{metrics[i % 4]}",
                "question": f"테스트 질문 {i}번",
                "target": f"정답 {i}",
                "candidate": f"후보 응답 {i}",
                "metric": metrics[i % 4],
            }
            for i in range(args.n)
        ]
        reqs = write_judge_batch_requests(items, judge_model=model)

    logger.info("built %d requests for task=%s model=%s", len(reqs), args.task, model)
    # Print a trimmed preview so the caller can eyeball structure.
    for r in reqs:
        preview = json.dumps(r, ensure_ascii=False, default=_json_default)
        if len(preview) > 2000:
            preview = preview[:2000] + "... <truncated>"
        print(preview)

    # NOTE: no submit_batch / poll_batch / collect_batch_results calls here —
    # actual batch submission must be triggered explicitly by caller code
    # that knows the run is intended. Smoke test is request-build only.
    logger.info("smoke test done — no batch submitted. Call submit_batch() "
                "from orchestrator code when ready.")


if __name__ == "__main__":
    _main()
