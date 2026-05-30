#!/usr/bin/env python3
"""Poll pending batches, download outputs, and aggregate into leaderboards.

Walks both batch registries written by ``submit_generation_batches.py`` and
``submit_judge_batches.py``, and for every registry entry whose status is not
yet terminal asks the matching provider module to poll / download / parse.

Per-model output files
----------------------
* Generation:
    results/phase5_exp_b_provider/expB__gemma-embed-300m__<model>.json
  (same shape as the legacy ``phase5_provider.py`` output so existing
  consumers like ``scripts/judge_leaderboard.py`` keep working.)

* Judge:
    results/phase5_judge_flagship/judge_<judge>__expB__gemma-embed-300m__<candidate>.json
  (mirrors the ``results/phase5_judge/judge_*.json`` shape: ``scores`` = list
  of per-question vote dicts with ``o_count``/``result`` fields.)

Leaderboard
-----------
After each poll cycle the script re-computes
``results/phase5_judge_flagship/leaderboard.json`` using the same schema as
``results/phase5_judge/leaderboard.json`` — ``{judges, excluded, ranking,
partial}``.

CLI
---
  # One-shot: check every pending batch once, download ready ones, then exit.
  python scripts/collect_and_aggregate.py --poll-once

  # Run in a loop until all batches reach a terminal state.
  python scripts/collect_and_aggregate.py --loop 60

  # Only collect generation batches (skip judges).
  python scripts/collect_and_aggregate.py --poll-once --only gen

Rules
-----
* Idempotent — re-entry is safe. Already-downloaded batches are not re-fetched
  if their output file already exists and covers all 300 qids.
* No ``max_tokens`` cap involvement — we only read outputs.
* Log lines use the same prefixed format as the submitter scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

ROOT = _SCRIPT_DIR.parent
GEN_OUT_DIR = ROOT / "results" / "phase5_exp_b_provider"
GEN_REGISTRY = GEN_OUT_DIR / "_batch_registry.json"
JUDGE_OUT_DIR = ROOT / "results" / "phase5_judge_flagship"
JUDGE_REGISTRY = JUDGE_OUT_DIR / "_batch_registry.json"
LEADERBOARD_PATH = JUDGE_OUT_DIR / "leaderboard.json"
GT_PATH = ROOT / "data" / "ground_truth_filtered.json"

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled", "error", "skipped"}
METRICS = ("similarity", "correctness", "completeness", "faithfulness")
THRESHOLD = 4  # allganize methodology, majority vote (≥2 of 4 metrics ≥ 4 → O)


# Broken judges flagged by the local pipeline heuristic (all-zero parser
# results). Populated if the scorer writes <5% non-zero across a file — see
# _aggregate_leaderboard.
EXCLUDE_JUDGES: set[str] = set()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _now_iso_minute() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M")


def log(provider: str, model: str, msg: str) -> None:
    print(f"[{_now_iso_minute()} {provider} {model}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------

def load_registry(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log("registry", path.name, "corrupt JSON — starting fresh")
    return {}


def save_registry(path: Path, reg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

def _poll_and_download(provider: str, batch_id: str, output_dir: Path) -> dict:
    """Return a dict describing the batch's terminal state and any downloaded
    file paths. Shape::

        {
          "state":       "completed|failed|running|unknown",
          "output_path": Path | None,     # only for openai (single combined file)
          "results_jsonl": Path | None,   # for anthropic
          "details":     {...},           # provider-specific
        }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if provider == "openai":
        from providers import openai_batch  # type: ignore
        # Best-effort single retrieval — for a non-terminal state we return
        # running so the caller can re-enter on the next poll tick.
        # openai_batch.poll_batch() blocks until terminal; we wrap a non-
        # blocking view by reusing its client directly.
        client = openai_batch._client()  # type: ignore[attr-defined]
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        if status not in {"completed", "failed", "expired", "cancelled"}:
            return {"state": "running", "details": {"status": status}}
        res = openai_batch.download_batch_output(batch_id, output_dir)
        return {
            "state": "completed" if status == "completed" else "failed",
            "output_path": res.get("output"),
            "details": res.get("stats", {}),
        }

    if provider == "anthropic":
        from providers import anthropic_batch  # type: ignore
        client = anthropic_batch._client()  # type: ignore[attr-defined]
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status != "ended":
            return {"state": "running", "details": {"status": b.processing_status}}
        results_path = output_dir / f"{batch_id}__results.jsonl"
        summary = anthropic_batch.collect_batch_results(batch_id, results_path)
        return {
            "state": "completed" if summary.get("errored", 0) == 0 else "failed",
            "results_jsonl": results_path,
            "details": summary,
        }

    if provider == "google":
        from providers import gemini_batch  # type: ignore
        client = gemini_batch._client()  # type: ignore[attr-defined]
        job = client.batches.get(name=batch_id)
        state_str = str(getattr(job, "state", "") or getattr(job, "status", ""))
        if not any(t in state_str for t in gemini_batch.TERMINAL_STATES):
            return {"state": "running", "details": {"status": state_str}}
        output_path = output_dir / f"{batch_id.replace('/', '_')}__output.jsonl"
        try:
            stats = gemini_batch.download_batch_output(batch_id, output_path)
        except Exception as exc:  # noqa: BLE001
            return {"state": "failed", "details": {"error": repr(exc)}}
        return {
            "state": "completed" if "SUCCEEDED" in state_str else "failed",
            "output_path": output_path,
            "details": stats,
        }

    if provider == "openrouter":
        # Realtime — already on disk when submit_generation_batches.py
        # returned. Nothing to poll.
        return {"state": "completed", "details": {"note": "realtime, no poll"}}

    return {"state": "unknown", "details": {}}


# ---------------------------------------------------------------------------
# Generation output parsers → unified shape
# ---------------------------------------------------------------------------

def _unified_generation_payload(
    model_id: str,
    provider: str,
    rows: list[dict],
    qid_to_row: dict,
    extra: dict | None = None,
) -> dict:
    """Build a file matching phase5_provider.json layout.

    rows: per-item list of {qid, generated_answer, input_tokens, output_tokens, latency_sec}
    qid_to_row: {qid: gt row from cache} so we can fill target/domain metadata.
    """
    total_in = sum(int(r.get("input_tokens") or 0) for r in rows)
    total_out = sum(int(r.get("output_tokens") or 0) for r in rows)
    answered = sum(1 for r in rows if r.get("generated_answer"))
    latencies = [r.get("latency_sec") for r in rows if r.get("latency_sec")]
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0
    body = {
        "provider": provider,
        "llm": model_id,
        "embed_model": "gemma-embed-300m",
        "top_k": 5,
        "total": len(rows),
        "answered": answered,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "avg_latency_sec": avg_lat,
        "results": rows,
    }
    if extra:
        body.update(extra)
    return body


def _parse_openai_generation(output_jsonl: Path, qid_to_row: dict) -> list[dict]:
    from providers import openai_batch  # type: ignore
    parsed = openai_batch.parse_generation_results(output_jsonl)
    out: list[dict] = []
    for rec in parsed:
        cid = rec.get("custom_id") or ""
        qid = cid.split("::", 1)[-1] if "::" in cid else cid
        gt = qid_to_row.get(qid, {})
        usage = rec.get("usage") or {}
        out.append({
            "qid": qid,
            "question": gt.get("question", ""),
            "target_answer": gt.get("target", ""),
            "domain": gt.get("domain", ""),
            "generated_answer": rec.get("content", ""),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "latency_sec": 0,
            "error": rec.get("error"),
        })
    return out


def _parse_anthropic_generation(results_jsonl: Path, qid_to_row: dict) -> list[dict]:
    from providers import anthropic_batch  # type: ignore
    parsed = anthropic_batch.parse_generation_results(results_jsonl)
    out: list[dict] = []
    for rec in parsed:
        qid = str(rec.get("custom_id") or "")
        gt = qid_to_row.get(qid, {})
        usage = rec.get("usage") or {}
        # Anthropic usage uses different token keys.
        in_tok = (
            usage.get("input_tokens")
            or 0
        ) + (usage.get("cache_read_input_tokens") or 0)
        out.append({
            "qid": qid,
            "question": gt.get("question", ""),
            "target_answer": gt.get("target", ""),
            "domain": gt.get("domain", ""),
            "generated_answer": rec.get("content", ""),
            "input_tokens": in_tok,
            "output_tokens": usage.get("output_tokens", 0),
            "latency_sec": 0,
            "error": rec.get("error"),
        })
    return out


def _parse_gemini_generation(output_jsonl: Path, qid_to_row: dict) -> list[dict]:
    from providers import gemini_batch  # type: ignore
    parsed = gemini_batch.parse_generation_results(output_jsonl)
    out: list[dict] = []
    for rec in parsed:
        qid = str(rec.get("qid") or "")
        gt = qid_to_row.get(qid, {})
        usage = rec.get("usage") or {}
        out.append({
            "qid": qid,
            "question": gt.get("question", ""),
            "target_answer": gt.get("target", ""),
            "domain": gt.get("domain", ""),
            "generated_answer": rec.get("text", ""),
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("candidates_tokens") or 0,
            "latency_sec": 0,
            "error": rec.get("error"),
        })
    return out


def _parse_openrouter_generation(jsonl_path: Path, qid_to_row: dict) -> list[dict]:
    """OpenRouter realtime already writes per-line JSON with {qid, answer, usage}."""
    out: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(rec.get("qid") or "")
            gt = qid_to_row.get(qid, {})
            usage = rec.get("usage") or {}
            out.append({
                "qid": qid,
                "question": rec.get("question") or gt.get("question", ""),
                "target_answer": gt.get("target", ""),
                "domain": gt.get("domain", ""),
                "generated_answer": rec.get("answer", ""),
                "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
                "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens") or 0,
                "latency_sec": (rec.get("latency_ms") or 0) / 1000.0,
                "error": rec.get("error"),
            })
    return out


# ---------------------------------------------------------------------------
# Judge output parsing → unified per-(judge, candidate) file
# ---------------------------------------------------------------------------

_CUSTOM_ID_RE = re.compile(r"^(?P<cand>.+)::(?P<qid>q\d{3})::(?P<metric>[a-z]+)$")


def _parse_and_write_judge_file(
    judge_id: str,
    candidate_id: str,
    provider: str,
    raw_path: Path,
    out_path: Path,
    total_questions: int,
) -> None:
    """Parse provider-specific judge raw output and write the unified file.

    Unified shape (same as results/phase5_judge/judge_*.json)::

      {
        "judge_model":   "<wire_model>",
        "judge_mode":    "flagship",
        "threshold":     4,
        "metrics":       ["similarity","correctness","completeness","faithfulness"],
        "original_file": "expB__gemma-embed-300m__<candidate>.json",
        "llm":           "<candidate_id>",
        "total":         300,
        "scored":        <count>,
        "o_count":       <count>,
        "x_count":       <count>,
        "accuracy":      <float>,
        "scores":        [ {"votes":{...}, "o_count":int, "result":"O"|"X"}, ... ]
      }
    """
    if provider == "openai":
        from providers import openai_batch  # type: ignore
        parsed = openai_batch.parse_judge_results(raw_path)
    elif provider == "anthropic":
        from providers import anthropic_batch  # type: ignore
        parsed = anthropic_batch.parse_judge_results(raw_path)
    elif provider == "google":
        from providers import gemini_batch  # type: ignore
        parsed = gemini_batch.parse_judge_results(raw_path)
    else:
        raise ValueError(f"unexpected provider for judge: {provider}")

    # Collate scores per (qid, metric).
    per_qid: dict[str, dict[str, int]] = defaultdict(dict)
    for rec in parsed:
        m = _CUSTOM_ID_RE.match(str(rec.get("custom_id") or ""))
        if not m:
            continue
        if m.group("cand") != candidate_id:
            # Not ours — skip.
            continue
        qid = m.group("qid")
        metric = m.group("metric")
        per_qid[qid][metric] = int(rec.get("score") or 0)

    rows: list[dict] = []
    o_count = 0
    x_count = 0
    scored = 0
    # Preserve qid ordering q000..q(total-1).
    for i in range(total_questions):
        qid = f"q{i:03d}"
        votes = per_qid.get(qid, {})
        if not votes:
            rows.append({"votes": {m: 0 for m in METRICS}, "o_count": 0, "result": "X", "missing": True})
            x_count += 1
            continue
        scored += 1
        oc = sum(1 for m in METRICS if (votes.get(m) or 0) >= THRESHOLD)
        result = "O" if oc >= 2 else "X"
        if result == "O":
            o_count += 1
        else:
            x_count += 1
        rows.append({"votes": {m: votes.get(m, 0) for m in METRICS}, "o_count": oc, "result": result})

    wire_model = {
        "gpt-5.4-pro":     "gpt-5.4-pro",
        "claude-opus-4-7": "claude-opus-4-7",
        "gemini-3.1-pro":  "gemini-3.1-pro",
    }.get(judge_id, judge_id)

    body = {
        "judge_model": wire_model,
        "judge_mode": "flagship",
        "threshold": THRESHOLD,
        "metrics": list(METRICS),
        "original_file": f"expB__gemma-embed-300m__{candidate_id}.json",
        "llm": candidate_id,
        "total": total_questions,
        "scored": scored,
        "o_count": o_count,
        "x_count": x_count,
        "accuracy": round(o_count / total_questions, 4) if total_questions else 0.0,
        "scores": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    log("judge", judge_id, f"[{candidate_id}] wrote {out_path.name} accuracy={body['accuracy']}")


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------

def _load_qid_map() -> dict[str, dict]:
    """Return {qid: {question,target,domain}} so generation parsers can reattach
    question/target metadata that the batch output rows do not carry."""
    if not GT_PATH.exists():
        return {}
    raw = json.loads(GT_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    if isinstance(raw, list):
        for i, r in enumerate(raw):
            out[f"q{i:03d}"] = {
                "question": r.get("question", ""),
                "target": r.get("target_answer", ""),
                "domain": r.get("domain", ""),
            }
    else:
        for k, v in raw.items():
            out[str(k)] = {
                "question": v.get("question", ""),
                "target": v.get("target", v.get("target_answer", "")),
                "domain": v.get("domain", ""),
            }
    return out


def process_generation_registry(reg: dict, qid_map: dict, dry_only: bool) -> bool:
    """Walk the generation registry; for any non-terminal entry, poll and try
    to download. Returns True iff at least one entry is still non-terminal
    after this pass (i.e. caller should keep looping).
    """
    any_pending = False
    for provider, models in reg.items():
        if not isinstance(models, dict):
            continue
        for model_id, entry in models.items():
            status = entry.get("status")
            output_path = Path(entry.get("output_path", ""))
            if status in TERMINAL_STATUSES:
                continue
            if output_path.exists() and status != "submitted":
                # Treat as completed.
                entry["status"] = "completed"
                continue
            if dry_only:
                log(provider, model_id, f"pending batch_id={entry.get('batch_id')} (dry-only)")
                any_pending = True
                continue
            try:
                res = _poll_and_download(
                    provider,
                    entry["batch_id"],
                    output_dir=GEN_OUT_DIR / "_raw" / provider,
                )
            except Exception as exc:  # noqa: BLE001
                log(provider, model_id, f"poll/download ERROR: {exc!r}")
                entry["status"] = "error"
                entry["error"] = repr(exc)
                continue

            if res["state"] == "running":
                log(provider, model_id, f"still running: {res.get('details')}")
                any_pending = True
                continue
            if res["state"] != "completed":
                log(provider, model_id, f"terminal but not completed: {res}")
                entry["status"] = "failed"
                continue

            # Parse provider output into unified shape.
            raw_file = res.get("output_path") or res.get("results_jsonl")
            if provider == "openrouter":
                raw_file = Path(entry.get("output_path", ""))
                rows = _parse_openrouter_generation(raw_file, qid_map) if raw_file.exists() else []
            elif provider == "openai":
                rows = _parse_openai_generation(Path(raw_file), qid_map)
            elif provider == "anthropic":
                rows = _parse_anthropic_generation(Path(raw_file), qid_map)
            elif provider == "google":
                rows = _parse_gemini_generation(Path(raw_file), qid_map)
            else:
                rows = []

            if provider != "openrouter":
                payload = _unified_generation_payload(model_id, provider, rows, qid_map)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            entry["status"] = "completed"
            log(provider, model_id, f"wrote {output_path.name} ({len(rows)} rows)")
    return any_pending


def process_judge_registry(reg: dict, dry_only: bool) -> bool:
    any_pending = False
    for judge_id, cand_map in reg.items():
        if not isinstance(cand_map, dict):
            continue
        provider = {
            "gpt-5.4-pro": "openai",
            "claude-opus-4-7": "anthropic",
            "gemini-3.1-pro": "google",
        }.get(judge_id)
        if provider is None:
            log("judge", judge_id, "unknown judge; skipping")
            continue
        for cand_id, entry in cand_map.items():
            status = entry.get("status")
            output_path = Path(entry.get("output_path", ""))
            if status in TERMINAL_STATUSES and output_path.exists():
                continue
            if output_path.exists() and status != "submitted":
                entry["status"] = "completed"
                continue
            if dry_only:
                log("judge", judge_id, f"[{cand_id}] pending batch_id={entry.get('batch_id')} (dry-only)")
                any_pending = True
                continue
            try:
                res = _poll_and_download(
                    provider, entry["batch_id"],
                    output_dir=JUDGE_OUT_DIR / "_raw" / provider,
                )
            except Exception as exc:  # noqa: BLE001
                log("judge", judge_id, f"[{cand_id}] poll/download ERROR: {exc!r}")
                entry["status"] = "error"
                entry["error"] = repr(exc)
                continue
            if res["state"] == "running":
                any_pending = True
                log("judge", judge_id, f"[{cand_id}] running: {res.get('details')}")
                continue
            if res["state"] != "completed":
                entry["status"] = "failed"
                log("judge", judge_id, f"[{cand_id}] not completed: {res}")
                continue

            raw_file = res.get("output_path") or res.get("results_jsonl")
            if raw_file is None:
                entry["status"] = "failed"
                continue
            try:
                _parse_and_write_judge_file(
                    judge_id=judge_id,
                    candidate_id=cand_id,
                    provider=provider,
                    raw_path=Path(raw_file),
                    out_path=output_path,
                    total_questions=300,
                )
                entry["status"] = "completed"
            except Exception as exc:  # noqa: BLE001
                log("judge", judge_id, f"[{cand_id}] parse ERROR: {exc!r}")
                entry["status"] = "error"
                entry["error"] = repr(exc)
    return any_pending


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def _aggregate_leaderboard() -> dict | None:
    files = sorted(JUDGE_OUT_DIR.glob("judge_*.json"))
    if not files:
        return None
    j: dict[str, dict[str, float]] = defaultdict(dict)
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if d.get("scored", 0) != d.get("total", 0):
            # Partial — still include; _compute_ranking will route it into partial[]
            pass
        judge = f"{d.get('judge_model')}_{d.get('judge_mode')}"
        if judge in EXCLUDE_JUDGES:
            continue
        m = re.match(r"judge_.+?__expB__gemma-embed-300m__(.+)\.json", f.name)
        if not m:
            continue
        llm = m.group(1)
        acc = d.get("accuracy")
        if acc is None:
            continue
        j[llm][judge] = acc

    judges = sorted({jn for d in j.values() for jn in d})
    complete_llms = [llm for llm, d in j.items() if all(jn in d for jn in judges)]

    ranking = []
    for llm in complete_llms:
        scores = {jn: j[llm][jn] for jn in judges}
        avg = sum(scores.values()) / len(scores)
        ranking.append({"llm": llm, "scores": scores, "avg": avg})
    ranking.sort(key=lambda r: -r["avg"])

    partial = {
        llm: {jn: j[llm][jn] for jn in judges if jn in j[llm]}
        for llm in j
        if llm not in complete_llms
    }
    return {
        "judges": judges,
        "excluded": sorted(EXCLUDE_JUDGES),
        "ranking": ranking,
        "partial": partial,
    }


def _write_leaderboard() -> None:
    lb = _aggregate_leaderboard()
    if not lb:
        return
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.write_text(
        json.dumps(lb, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log(
        "leaderboard", "_",
        f"wrote {LEADERBOARD_PATH.name}: "
        f"{len(lb['ranking'])} complete / {len(lb['partial'])} partial / "
        f"{len(lb['judges'])} judges",
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--poll-once", action="store_true", help="Single pass, then exit.")
    group.add_argument(
        "--loop", type=int, default=0, metavar="SECONDS",
        help="Poll every SECONDS until all batches terminal (0 = one-shot).",
    )
    parser.add_argument(
        "--only", choices=("gen", "judge"), default=None,
        help="Restrict processing to generation or judge registries.",
    )
    parser.add_argument(
        "--dry-only", action="store_true",
        help="Do not poll APIs; just list pending entries and leave them alone.",
    )
    args = parser.parse_args(argv)

    qid_map = _load_qid_map()

    def _one_pass() -> bool:
        pending = False
        if args.only in (None, "gen") and GEN_REGISTRY.exists():
            gen_reg = load_registry(GEN_REGISTRY)
            pending |= process_generation_registry(gen_reg, qid_map, dry_only=args.dry_only)
            save_registry(GEN_REGISTRY, gen_reg)
        if args.only in (None, "judge") and JUDGE_REGISTRY.exists():
            judge_reg = load_registry(JUDGE_REGISTRY)
            pending |= process_judge_registry(judge_reg, dry_only=args.dry_only)
            save_registry(JUDGE_REGISTRY, judge_reg)
        _write_leaderboard()
        return pending

    if args.loop and args.loop > 0:
        while True:
            still_pending = _one_pass()
            if not still_pending:
                log("orchestrator", "_", "all registries terminal; exiting loop")
                return 0
            time.sleep(args.loop)
    else:
        _one_pass()
        return 0


if __name__ == "__main__":
    sys.exit(main())
