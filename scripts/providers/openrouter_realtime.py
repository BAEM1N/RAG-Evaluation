#!/usr/bin/env python3
"""
OpenRouter realtime generation provider.

Purpose
-------
OpenRouter is used to fill gaps in the Phase-5 generation grid for models that
the three direct providers (OpenAI, Anthropic, Google) do not ship themselves:

    - xai/grok-4.20
    - moonshotai/kimi-k2.5
    - minimax/minimax-m2.7
    - qwen/qwen3-max-thinking
    - qwen/qwen3.6-plus
    - z-ai/glm-5.1
    - perplexity/sonar-reasoning-pro
    - cohere/command-a
    - mistralai/mistral-large-3

OpenRouter has no batch API, so this module performs realtime chat completions
concurrently with a ThreadPoolExecutor. Judge/scoring is intentionally NOT
implemented here — flagship judges (GPT-5.4 Pro, Claude Opus 4.7, Gemini 3.1
Pro) are driven through their direct SDK modules, not through OpenRouter.

Hard rules (do NOT violate):
    - Never set ``max_tokens`` / ``num_predict`` / any output cap.
    - Use the ``openai`` SDK (>=2.31) pointed at
      ``https://openrouter.ai/api/v1`` with the ``OPENROUTER_API_KEY`` env var.
    - Attach Anthropic-style ``cache_control: {"type": "ephemeral"}`` blocks on
      the static prefix; OpenAI/Gemini-routed models ignore it safely, and
      Anthropic-routed models benefit directly. OpenRouter's sticky routing
      then maximises prompt-cache hits across concurrent workers.
    - Concurrency is fixed to ``ThreadPoolExecutor(max_workers=8)`` by default.
    - Retry 3x on 429/5xx with exponential backoff (2**n seconds).
    - JSONL output is appended after each success for resumability.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

try:
    from openai import OpenAI
    from openai import APIStatusError, APIConnectionError, RateLimitError
except ImportError as e:  # pragma: no cover - import guard
    raise SystemExit(
        "openai>=2.31 is required: `uv add openai` or `pip install openai`"
    ) from e


# ─────────────────────────── logging / constants ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("openrouter-realtime")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Any prompt whose total static text exceeds this many characters is deemed
# "large enough" to benefit from ephemeral cache markers (Anthropic-routed
# models require ≥1024 tokens to cache; ~3.5 chars/token → 4000 chars).
CACHE_MIN_CHARS = 4000

# How many retries on 429 / 5xx / connection errors.
MAX_RETRIES = 3

# Referer / X-Title help OpenRouter's routing analytics — optional.
DEFAULT_HEADERS = {
    "HTTP-Referer": os.environ.get(
        "OPENROUTER_REFERER", "https://github.com/baem1n/RAG-Evaluation"
    ),
    "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "RAG-Evaluation Phase 5"),
}

SYSTEM_PROMPT = (
    "You are a careful retrieval-augmented assistant. "
    "Use ONLY the supplied context to answer the user's question. "
    "If the context is insufficient, say you do not know. "
    "Keep the answer concise — at most three sentences."
)

# Lock serialising appends to the JSONL output file across workers.
_write_lock = threading.Lock()


# ───────────────────────────── client factory ─────────────────────────────
def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before running this module."
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers=DEFAULT_HEADERS,
    )


# ─────────────────────────── message builder ────────────────────────────
def build_generation_messages(
    question: str,
    context: list[str],
    enable_cache: bool = True,
) -> list[dict]:
    """Build the chat-completions messages array.

    The system + context block is placed as the cacheable prefix. For models
    routed through Anthropic, we attach ``cache_control={"type":"ephemeral"}``
    to each content part in the system message (OpenRouter forwards this
    unchanged). OpenAI- and Gemini-routed models simply ignore the extra
    field — prompt caching there is automatic / implicit.

    If ``enable_cache`` is False, or the static prefix is below
    ``CACHE_MIN_CHARS``, we emit a plain string system message for maximum
    compatibility with exotic providers.
    """
    if not isinstance(context, list):
        raise TypeError("context must be a list[str]")

    context_text = "\n\n".join(
        f"[doc {i + 1}]\n{chunk}" for i, chunk in enumerate(context)
    )
    static_prefix = f"{SYSTEM_PROMPT}\n\n=== CONTEXT ===\n{context_text}"

    use_structured_cache = enable_cache and len(static_prefix) >= CACHE_MIN_CHARS

    if use_structured_cache:
        system_message: dict[str, Any] = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                },
                {
                    "type": "text",
                    "text": f"=== CONTEXT ===\n{context_text}",
                    # Anthropic-routed models honour this; others ignore it.
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }
    else:
        system_message = {"role": "system", "content": static_prefix}

    user_message = {
        "role": "user",
        "content": f"Question: {question}\n\nAnswer:",
    }

    return [system_message, user_message]


# ────────────────────────── low-level request ──────────────────────────
def _should_retry(exc: Exception) -> bool:
    """True if the exception is retriable (429, 5xx, or connection error)."""
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status is None:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
        return status == 429 or (isinstance(status, int) and status >= 500)
    return False


def _call_once(
    client: OpenAI,
    model: str,
    messages: list[dict],
) -> dict:
    """Single OpenRouter call with retry on 429/5xx.

    Returns the raw SDK response converted via ``model_dump()``.
    Never passes ``max_tokens`` / ``max_completion_tokens`` — per the project
    rule banning any output cap.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                # IMPORTANT: no max_tokens / num_predict — forbidden by spec.
            )
            return resp.model_dump()
        except Exception as exc:  # noqa: BLE001 — retry gate below
            last_exc = exc
            if attempt >= MAX_RETRIES or not _should_retry(exc):
                raise
            # Exponential backoff with jitter: 2, 4, 8 seconds.
            sleep_s = (2 ** (attempt + 1)) + random.uniform(0, 0.5)
            logger.warning(
                "retry %d/%d after %.1fs for model=%s: %s",
                attempt + 1,
                MAX_RETRIES,
                sleep_s,
                model,
                exc,
            )
            time.sleep(sleep_s)
    # Unreachable — loop either returns or raises — but keep linters happy.
    raise last_exc if last_exc else RuntimeError("unknown error")


# ────────────────────────────── workers ───────────────────────────────
def _extract_answer(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _process_item(
    client: OpenAI,
    model: str,
    item: dict,
    enable_cache: bool,
) -> dict:
    qid = item["qid"]
    messages = build_generation_messages(
        question=item["question"],
        context=item.get("context", []),
        enable_cache=enable_cache,
    )
    t0 = time.perf_counter()
    resp = _call_once(client, model, messages)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    usage = resp.get("usage") or {}
    # OpenRouter exposes prompt-cache savings as `usage.cache_discount`
    # (negative = credits refunded). Surface it verbatim when present.
    cache_discount = usage.get("cache_discount")

    return {
        "qid": qid,
        "model": model,
        "question": item["question"],
        "answer": _extract_answer(resp),
        "usage": usage,
        "cache_discount": cache_discount,
        "latency_ms": round(latency_ms, 2),
        "provider_meta": {
            "id": resp.get("id"),
            "provider": resp.get("provider"),
            "openrouter_model": resp.get("model"),
        },
    }


def _append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ───────────────────────────── public API ─────────────────────────────
def run_generation(
    items: list[dict],
    model: str,
    out_path: Path,
    parallel: int = 8,
    enable_cache: bool = True,
) -> dict:
    """Run OpenRouter chat completions over ``items`` with a thread pool.

    Each item must be shaped ``{'qid', 'question', 'context': list[str]}``.
    Results are appended to ``out_path`` as JSONL after each success, so the
    file itself doubles as the resume checkpoint.

    Returns aggregate stats::

        {
            'n_ok': int,
            'n_err': int,
            'cache_discount_total': float,
            'total_tokens': int,
        }
    """
    if parallel < 1:
        raise ValueError("parallel must be >= 1")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    n_ok = 0
    n_err = 0
    cache_discount_total = 0.0
    total_tokens = 0

    logger.info(
        "run_generation: model=%s items=%d parallel=%d out=%s",
        model,
        len(items),
        parallel,
        out_path,
    )

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        future_to_item = {
            pool.submit(_process_item, client, model, item, enable_cache): item
            for item in items
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            qid = item.get("qid", "<unknown>")
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 — per-item isolation
                n_err += 1
                logger.error("qid=%s failed: %s", qid, exc)
                _append_jsonl(
                    out_path,
                    {
                        "qid": qid,
                        "model": model,
                        "error": repr(exc),
                    },
                )
                continue

            n_ok += 1
            usage = record.get("usage") or {}
            total_tokens += int(usage.get("total_tokens") or 0)
            discount = record.get("cache_discount")
            if isinstance(discount, (int, float)):
                cache_discount_total += float(discount)
            _append_jsonl(out_path, record)

    stats = {
        "n_ok": n_ok,
        "n_err": n_err,
        "cache_discount_total": cache_discount_total,
        "total_tokens": total_tokens,
    }
    logger.info("run_generation done: %s", stats)
    return stats


def _load_completed_qids(path: Path) -> set[str]:
    """Return the set of qids already present as OK (non-error) rows."""
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in row:
                continue
            qid = row.get("qid")
            if qid is not None:
                completed.add(qid)
    return completed


def resume_generation(
    items: list[dict],
    model: str,
    out_path: Path,
    parallel: int = 8,
    enable_cache: bool = True,
) -> dict:
    """Skip qids already persisted in ``out_path``, then delegate to
    :func:`run_generation` for the remainder.
    """
    out_path = Path(out_path)
    completed = _load_completed_qids(out_path)
    remaining = [it for it in items if it.get("qid") not in completed]
    logger.info(
        "resume_generation: total=%d completed=%d remaining=%d out=%s",
        len(items),
        len(completed),
        len(remaining),
        out_path,
    )
    if not remaining:
        return {
            "n_ok": 0,
            "n_err": 0,
            "cache_discount_total": 0.0,
            "total_tokens": 0,
        }
    return run_generation(
        remaining,
        model=model,
        out_path=out_path,
        parallel=parallel,
        enable_cache=enable_cache,
    )


# ────────────────────────────── CLI helpers ──────────────────────────────
def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenRouter realtime generation for RAG-Evaluation Phase 5",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="OpenRouter model slug, e.g. xai/grok-4.20, qwen/qwen3-max-thinking",
    )
    parser.add_argument(
        "--items",
        required=True,
        type=Path,
        help="Path to JSONL of items [{qid, question, context:[...]}].",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to output JSONL (appended, resumable).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="ThreadPoolExecutor max_workers (default: 8).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable ephemeral cache_control on the static prefix.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing --out contents and re-run every qid.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.items.exists():
        logger.error("items file not found: %s", args.items)
        return 2
    items = list(_iter_jsonl(args.items))
    logger.info("loaded %d items from %s", len(items), args.items)

    runner = run_generation if args.no_resume else resume_generation
    stats = runner(
        items=items,
        model=args.model,
        out_path=args.out,
        parallel=args.parallel,
        enable_cache=not args.no_cache,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats["n_err"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
