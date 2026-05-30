#!/usr/bin/env python3
"""Reranker GPU bench — chunks/top20 캐시 기반, AMD GPU에서 실행.

사용 (HP Z2 Mini, AI 395+ ROCm):
  source ~/rag-bench/.venv/bin/activate
  cd ~/rag-bench/repo
  python scripts/bench_reranker_gpu.py --rerankers all --skip-existing

캐시 파일 (JSON + npy):
  results/phase4_2_reranker/_chunks_cache.json  — chunks/queries/meta/gt
  results/phase4_2_reranker/_retrieved_top20.npy — int64 array (300, 20)
"""
import argparse, json, time, sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # flash/mem attention on AMD
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "results/phase4_2_reranker"
CACHE_JSON = OUT_DIR / "_chunks_cache.json"
CACHE_TOP20 = OUT_DIR / "_retrieved_top20.npy"
TOP_K_FINAL = 5
TOP_N_RETRIEVE = 20

# 28 reranker (기존 11 + 신규 17)
RERANKERS = {
    # 기존 (M5 Max 측정 완료, GPU 재측정으로 통일)
    "bge-reranker-v2-m3":     "BAAI/bge-reranker-v2-m3",
    "bge-reranker-v2-m3-ko":  "dragonkue/bge-reranker-v2-m3-ko",
    "qwen3-reranker-0.6b":    "Qwen/Qwen3-Reranker-0.6B",
    "qwen3-reranker-4b":      "Qwen/Qwen3-Reranker-4B",
    "mxbai-rerank-base-v2":   "mixedbread-ai/mxbai-rerank-base-v2",
    "mxbai-rerank-large-v2":  "mixedbread-ai/mxbai-rerank-large-v2",
    "bge-reranker-v2-gemma":  "BAAI/bge-reranker-v2-gemma",
    "ko-reranker":            "Dongjin-kr/ko-reranker",
    "modernReranker":         "naver/modernReranker",
    "pixie-spell-reranker":   "telepix/PIXIE-Spell-Reranker-Preview-0.6B",

    # ── 신규: Korean fine-tune ─────────────────────────────────────────
    "upskyy-ko-reranker":         "upskyy/ko-reranker",
    "upskyy-ko-reranker-8k":      "upskyy/ko-reranker-8k",
    "sigridjineth-ko-reranker-v1.1": "sigridjineth/ko-reranker-v1.1",
    "sigridjineth-ko-reranker-v1":   "sigridjineth/ko-reranker-v1",
    "mncai-bge-ko-reranker-560m": "mncai/bge-ko-reranker-560M",
    "kkresearch-bge-ko-finance":  "kkresearch/bge-reranker-v2-m3-korean-finance",
    "shoxa-mir-bge-v2-m3-ko":     "shoxa-mir/bge-reranker-v2-m3-ko",
    "ktds-vue-code-search-ko":    "SeoJHeasdw/ktds-vue-code-search-reranker-ko",
    "js2jang-ko-qnli":            "js2jang/reranker_ko_qnli",
    "naver-xprovence-bgem3-v1":   "naver/xprovence-reranker-bgem3-v1",
    "naver-xprovence-bgem3-v2":   "naver/xprovence-reranker-bgem3-v2",

    # ── 신규: BGE legacy / 신모델 ───────────────────────────────────────
    "bge-reranker-base":            "BAAI/bge-reranker-base",
    "bge-reranker-large":           "BAAI/bge-reranker-large",
    "bge-reranker-v2-minicpm":      "BAAI/bge-reranker-v2-minicpm-layerwise",
    "bge-reranker-v2.5-gemma2-lw":  "BAAI/bge-reranker-v2.5-gemma2-lightweight",

    # ── 신규: Jina 재시도 ──────────────────────────────────────────────
    "jina-reranker-v3":              "jinaai/jina-reranker-v3",
    "jina-reranker-v2-multilingual": "jinaai/jina-reranker-v2-base-multilingual",
    "jina-reranker-m0":              "jinaai/jina-reranker-m0",

    # ── 신규: 기타 ─────────────────────────────────────────────────────
    "gte-reranker-modernbert":   "Alibaba-NLP/gte-reranker-modernbert-base",
    "nvidia-llama-nemotron-1b":  "nvidia/llama-nemotron-rerank-1b-v2",
}


def load_cache():
    d = json.load(open(CACHE_JSON, encoding="utf-8"))
    top20 = np.load(CACHE_TOP20)
    return d["chunks_text"], d["meta"], d["queries"], d["gt"], top20


def rerank_with(model_id, chunks_text, queries, top20):
    from sentence_transformers import CrossEncoder
    print(f"  loading {model_id}...", flush=True)
    t0 = time.time()
    ce = CrossEncoder(model_id, trust_remote_code=True, device="cuda")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    n_q = len(queries)
    pairs = []
    origin = []
    for i in range(n_q):
        for r in range(TOP_N_RETRIEVE):
            idx = int(top20[i, r])
            pairs.append([queries[i], chunks_text[idx]])
            origin.append((i, idx))

    print(f"  scoring {len(pairs)} pairs (batch=32)...", flush=True)
    t = time.time()
    scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
    elapsed = time.time() - t
    print(f"  rerank done in {elapsed:.1f}s ({len(pairs)/elapsed:.0f} pairs/s)", flush=True)

    per_q_scores = [[] for _ in range(n_q)]
    per_q_idx = [[] for _ in range(n_q)]
    for (i, idx), s in zip(origin, scores):
        per_q_scores[i].append(float(s))
        per_q_idx[i].append(idx)

    out = np.zeros((n_q, TOP_K_FINAL), dtype=np.int64)
    for i in range(n_q):
        order = np.argsort(-np.array(per_q_scores[i]))[:TOP_K_FINAL]
        out[i] = [per_q_idx[i][j] for j in order]
    return out, elapsed


def compute_metrics(topk_idx, meta, gt):
    hit1 = hit5 = file_hit5 = 0
    mrr = 0.0
    for i, g in enumerate(gt):
        tgt_f = g["target_file_name"]
        try:
            tgt_p = int(str(g["target_page_no"]).strip().split(",")[0].strip())
        except (ValueError, AttributeError):
            tgt_p = None
        found = None; ff = False
        for rank, idx in enumerate(topk_idx[i]):
            m = meta[idx]
            if m["file"] == tgt_f:
                ff = True
                if tgt_p is None or m["page"] == tgt_p:
                    if found is None: found = rank + 1
        if found == 1: hit1 += 1
        if found and found <= 5: hit5 += 1
        if ff: file_hit5 += 1
        if found: mrr += 1.0 / found
    n = len(gt)
    return {"MRR": mrr/n, "Hit@1": hit1/n, "Hit@5": hit5/n, "File@5": file_hit5/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerankers", nargs="+", default=["all"])
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    chunks_text, meta, queries, gt, top20 = load_cache()
    print(f"cache: {len(chunks_text)} chunks, {len(queries)} queries, top20 {top20.shape}")

    keys = list(RERANKERS.keys()) if args.rerankers == ["all"] else args.rerankers

    results = {}
    for k in keys:
        if k not in RERANKERS:
            print(f"unknown: {k}"); continue
        out_path = OUT_DIR / f"{k}.json"
        if args.skip_existing and out_path.exists():
            print(f"\nSKIP {k} (cached)")
            results[k] = json.load(open(out_path)); continue
        print(f"\n=== {k} ===")
        try:
            topk, elapsed = rerank_with(RERANKERS[k], chunks_text, queries, top20)
            m = compute_metrics(topk, meta, gt)
            r = {"strategy": k, "model": RERANKERS[k], "rerank_time_sec": elapsed,
                 "device": "amd_gpu", **m}
            results[k] = r
            json.dump(r, open(out_path, "w"), ensure_ascii=False, indent=2)
            print(f"  MRR={m['MRR']:.4f} Hit@1={m['Hit@1']:.3f} Hit@5={m['Hit@5']:.3f} File@5={m['File@5']:.3f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERR {k}: {e}")

    print("\n" + "=" * 110)
    print(f"{'Strategy':<32} {'MRR':>7} {'Hit@1':>7} {'Hit@5':>7} {'File@5':>7} {'Time(s)':>8}")
    print("-" * 110)
    for k in sorted(results.keys(), key=lambda x: results[x].get("MRR", -1), reverse=True):
        r = results[k]
        print(f"{k:<32} {r['MRR']:>7.4f} {r['Hit@1']:>7.3f} {r['Hit@5']:>7.3f} {r['File@5']:>7.3f} {r.get('rerank_time_sec',0):>8.1f}")


if __name__ == "__main__":
    main()
