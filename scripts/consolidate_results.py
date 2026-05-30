#!/usr/bin/env python3
"""모든 stage 결과를 통합 CSV/JSON으로 정리.

산출:
  results/all_stages_summary.csv  — long-format wide table
  results/all_stages_summary.json — same data JSON
"""
import json, csv, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
OUT_CSV = RES / "all_stages_summary.csv"
OUT_JSON = RES / "all_stages_summary.json"


def safe(d, k, default=None):
    v = d.get(k, default)
    if isinstance(v, dict):
        return v
    return v


def from_json(path, defaults):
    try:
        d = json.load(open(path, encoding="utf-8"))
        return d
    except Exception as e:
        return None


def collect():
    rows = []

    # Stage 1: Loader
    for f in (RES / "phase1_parser_extended").glob("*.json"):
        d = from_json(f, {})
        rows.append({
            "stage": "1_loader",
            "config": d.get("loader") or f.stem,
            "MRR": d.get("MRR"),
            "Hit@1": d.get("Hit@1"),
            "Hit@5": d.get("Hit@5"),
            "File@5": d.get("File@5"),
            "n_chunks": d.get("n_chunks"),
            "parse_time_sec": d.get("parse_time_sec"),
            "fixed_pipeline": "chunking=1000/200, embed=gemma-300m, dense top-5",
        })

    # Stage 2: Parser (char-based extended)
    for f in (RES / "phase2_parser_extended").glob("*.json"):
        d = from_json(f, {})
        rows.append({
            "stage": "2_parser_char",
            "config": d.get("strategy") or f.stem,
            "label": d.get("label"),
            "MRR": d.get("MRR"),
            "Hit@1": d.get("Hit@1"),
            "Hit@5": d.get("Hit@5"),
            "File@5": d.get("File@5"),
            "n_chunks": d.get("n_chunks"),
            "parse_time_sec": d.get("parse_time_sec"),
            "fixed_pipeline": "loader=pymupdf, embed=gemma-300m, dense top-5",
        })

    # Stage 2-ext: Semantic + LLM-based
    for f in (RES / "phase2_parser_semantic").glob("*.json"):
        d = from_json(f, {})
        rows.append({
            "stage": "2_parser_semantic",
            "config": d.get("strategy") or f.stem,
            "label": d.get("label"),
            "MRR": d.get("MRR"),
            "Hit@1": d.get("Hit@1"),
            "Hit@5": d.get("Hit@5"),
            "File@5": d.get("File@5"),
            "n_chunks": d.get("n_chunks"),
            "parse_time_sec": d.get("parse_time_sec"),
            "fixed_pipeline": "loader=pymupdf, embed=gemma-300m, hybrid 3:7",
        })

    # Stage 3: Embedding (different JSON structure)
    for f in (RES / "phase4_embedding").glob("*.json"):
        d = from_json(f, {})
        m = d.get("metrics", d) if isinstance(d.get("metrics"), dict) else d
        rows.append({
            "stage": "3_embedding",
            "config": d.get("model") or d.get("name") or f.stem,
            "dim": d.get("dim"),
            "MRR": m.get("mrr") or m.get("MRR"),
            "Hit@1": m.get("page_hit@1") or m.get("Hit@1"),
            "Hit@5": m.get("page_hit@5") or m.get("Hit@5"),
            "File@5": m.get("file_hit@5") or m.get("File@5"),
            "fixed_pipeline": "parser=pymupdf4llm 500/100, FAISS, top-5",
        })

    # Stage 4: Retriever
    for f in (RES / "phase4_retriever_extended").glob("*.json"):
        d = from_json(f, {})
        rows.append({
            "stage": "4_retriever",
            "config": d.get("strategy") or f.stem,
            "MRR": d.get("MRR"),
            "Hit@1": d.get("Hit@1"),
            "Hit@5": d.get("Hit@5"),
            "File@5": d.get("File@5"),
            "n_chunks": d.get("n_chunks"),
            "fixed_pipeline": "loader=pymupdf, chunk=300/50, embed=gemma-300m",
        })

    # Stage 4-1: Pre-retriever
    for f in (RES / "phase4_1_pre_retriever").glob("*.json"):
        d = from_json(f, {})
        rows.append({
            "stage": "4-1_pre_retriever",
            "config": d.get("strategy") or f.stem,
            "MRR": d.get("MRR"),
            "Hit@1": d.get("Hit@1"),
            "Hit@5": d.get("Hit@5"),
            "File@5": d.get("File@5"),
            "llm_calls": d.get("llm_calls"),
            "elapsed_sec": d.get("elapsed_sec"),
            "fixed_pipeline": "Hybrid 3:7 + Stage 4-2 winner reranker, LLM=GPT-5.4",
        })

    # Stage 4-2: Post-retriever (Reranker)
    for f in (RES / "phase4_2_reranker").glob("*.json"):
        if "_chunks_cache" in f.name or "_retrieved" in f.name:
            continue
        d = from_json(f, {})
        rows.append({
            "stage": "4-2_post_retriever",
            "config": d.get("strategy") or f.stem,
            "model": d.get("model"),
            "MRR": d.get("MRR"),
            "Hit@1": d.get("Hit@1"),
            "Hit@5": d.get("Hit@5"),
            "File@5": d.get("File@5"),
            "rerank_time_sec": d.get("rerank_time_sec"),
            "device": d.get("device"),
            "fixed_pipeline": "PreR=baseline, Hybrid 3:7, top-20 → rerank → top-5",
        })

    # Stage 5 axis-wise (e2e)
    for f in (RES / "phase_e2e_axis_wise").glob("*.json"):
        d = from_json(f, {})
        rows.append({
            "stage": "5_e2e_axis_wise",
            "axis": d.get("axis"),
            "config": d.get("config") or f.stem,
            "prer": d.get("prer"),
            "r": d.get("r"),
            "postr": d.get("postr"),
            "MRR": (d.get("retrieval") or {}).get("MRR"),
            "Hit@1": (d.get("retrieval") or {}).get("Hit@1"),
            "Hit@5": (d.get("retrieval") or {}).get("Hit@5"),
            "File@5": (d.get("retrieval") or {}).get("File@5"),
            "judge_similarity": (d.get("judge_means") or {}).get("similarity"),
            "judge_correctness": (d.get("judge_means") or {}).get("correctness"),
            "judge_completeness": (d.get("judge_means") or {}).get("completeness"),
            "judge_faithfulness": (d.get("judge_means") or {}).get("faithfulness"),
            "judge_mean": d.get("judge_overall_mean"),
            "fixed_pipeline": "Stage 1-4-2 winners fixed except axis variable",
        })

    # Stage 5 Generation: 46 generation models leaderboard (from HF consolidated.parquet)
    try:
        import pandas as pd
        cons_path = Path("/tmp/Korean-RAG-LLM-Judge-Benchmark/data/consolidated.parquet")
        if cons_path.exists():
            df = pd.read_parquet(cons_path)
            ox_cols = [c for c in df.columns if c.endswith("_ox")]
            for c in ox_cols:
                model = c.replace("_ox", "")
                acc = (df[c] == "O").mean()
                weights = "open" if any(t in model.lower() for t in
                    ["qwen","deepseek","exaone","lfm","phi","minimax","mistral","moonshot","kimi","nemotron","gpt-oss","mimo","glm","supergemma","gemma4","solar-open"]) else "closed"
                rows.append({
                    "stage": "5_generation",
                    "config": model,
                    "weights": weights,
                    "accuracy": round(float(acc), 4),
                    "fixed_pipeline": "embed=gemma-300m, dense retrieval + ko-reranker (Phase 5 fixed pipeline), 18-judge consolidated (4-metric majority O)",
                })
    except ImportError:
        pass

    # Stage 6 Cartesian (384 configs)
    cart_path = RES / "cartesian/cartesian_summary.json"
    if cart_path.exists():
        cart = json.load(open(cart_path))
        for d in cart:
            rows.append({
                "stage": "6_cartesian",
                "config": d.get("config"),
                "prer": d.get("prer"),
                "r": d.get("r"),
                "postr": d.get("postr"),
                "MRR": d.get("MRR"),
                "Hit@1": d.get("Hit@1"),
                "Hit@5": d.get("Hit@5"),
                "File@5": d.get("File@5"),
                "judge_similarity": d.get("judge_similarity"),
                "judge_correctness": d.get("judge_correctness"),
                "judge_completeness": d.get("judge_completeness"),
                "judge_faithfulness": d.get("judge_faithfulness"),
                "judge_mean": d.get("judge_mean"),
                "accuracy": d.get("accuracy"),
                "fixed_pipeline": "loader=pymupdf, chunk=300/50, embed=gemma-300m, all 384 configs of PreR×R×PostR; accuracy = 4-metric majority O (Phase 5 동일 규칙)",
            })

    return rows


def main():
    rows = collect()
    print(f"Collected {len(rows)} rows across all stages")

    # Stage 별 카운트
    from collections import Counter
    cnt = Counter(r["stage"] for r in rows)
    for stage, n in sorted(cnt.items()):
        print(f"  {stage}: {n}")

    # Determine all columns (union of all keys)
    all_keys = ["stage", "config", "axis", "prer", "r", "postr", "label", "model", "device", "dim", "weights",
                "MRR", "Hit@1", "Hit@5", "File@5", "accuracy",
                "judge_similarity", "judge_correctness", "judge_completeness", "judge_faithfulness", "judge_mean",
                "n_chunks", "parse_time_sec", "rerank_time_sec", "llm_calls", "elapsed_sec",
                "fixed_pipeline"]

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_keys})
    print(f"\n✅ CSV: {OUT_CSV} ({OUT_CSV.stat().st_size/1024:.1f} KB)")

    # Write JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON: {OUT_JSON} ({OUT_JSON.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
