#!/usr/bin/env python3
"""Step 2: apply reranker to all (PreR, R) retrieval results for each PostR.

각 PostR 모델별로:
  - 모델 1회 로드
  - 모든 retrieval/top20_<prer>__<r>.npy 에 대해 rerank
  - top-5 저장: results/cartesian/topk/<prer>__<r>__<postr>.npy (shape 300×5)

디바이스 지정:
  - --device m5   → 모든 M5-routed PostR
  - --device amd  → 모든 AMD-routed PostR  (HP Z2 Mini에서 실행)
  - 또는 --rerankers <key1> <key2> ...

Usage (HP Z2 Mini):
  ssh baeumai@localhost
  cd ~/rag-bench/repo && source ~/rag-bench/.venv/bin/activate
  python scripts/cartesian/rerank_apply.py --device amd --skip-existing

Usage (M5):
  python scripts/cartesian/rerank_apply.py --device m5 --skip-existing
"""
import argparse, json, time, sys, os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cartesian.config import (
    PRER_STRATEGIES, R_STRATEGIES, POSTR_MODELS, m5_postr, amd_postr,
    TOP_N_RERANK, TOP_K_FINAL,
)

CHUNK_CACHE = ROOT / "results/phase4_2_reranker/_chunks_cache.json"
RETRIEVAL_DIR = ROOT / "results/cartesian/retrieval"
TOPK_DIR = ROOT / "results/cartesian/topk"
TOPK_DIR.mkdir(parents=True, exist_ok=True)


def load_cache():
    d = json.load(open(CHUNK_CACHE, encoding="utf-8"))
    return d["chunks_text"], d["queries"]


def rerank_one_combo(ce, chunks_text, queries, top20):
    n_q = len(queries)
    pairs = []
    origin = []
    for i in range(n_q):
        for r in range(TOP_N_RERANK):
            idx = int(top20[i, r])
            pairs.append([queries[i], chunks_text[idx]])
            origin.append((i, idx))
    scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
    per_q_s = [[] for _ in range(n_q)]
    per_q_i = [[] for _ in range(n_q)]
    for (i, idx), s in zip(origin, scores):
        per_q_s[i].append(float(s))
        per_q_i[i].append(idx)
    out = np.zeros((n_q, TOP_K_FINAL), dtype=np.int64)
    for i in range(n_q):
        order = np.argsort(-np.array(per_q_s[i]))[:TOP_K_FINAL]
        out[i] = [per_q_i[i][j] for j in order]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["m5", "amd"], help="select all rerankers for this device")
    ap.add_argument("--rerankers", nargs="+", help="explicit reranker keys")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if args.device == "m5":
        keys = m5_postr()
    elif args.device == "amd":
        keys = amd_postr()
    elif args.rerankers:
        keys = args.rerankers
    else:
        keys = list(POSTR_MODELS.keys())

    # no_rerank handled separately (just take top-5 of top-20)
    keys = [k for k in keys if k != "no_rerank"]

    chunks_text, queries = load_cache()
    print(f"chunks: {len(chunks_text)}, queries: {len(queries)}")

    # also handle no_rerank baseline once
    for prer in PRER_STRATEGIES:
        for r in R_STRATEGIES:
            in_path = RETRIEVAL_DIR / f"top20_{prer}__{r}.npy"
            out_path = TOPK_DIR / f"{prer}__{r}__no_rerank.npy"
            if args.skip_existing and out_path.exists():
                continue
            if not in_path.exists():
                continue
            top20 = np.load(in_path)
            np.save(out_path, top20[:, :TOP_K_FINAL])

    from sentence_transformers import CrossEncoder

    for k in keys:
        model_id, _ = POSTR_MODELS[k]
        if model_id is None: continue
        print(f"\n=== {k} ({model_id}) ===")
        t = time.time()
        ce = CrossEncoder(model_id, trust_remote_code=True, device="cuda" if args.device == "amd" else None)
        print(f"  loaded in {time.time()-t:.1f}s", flush=True)

        for prer in PRER_STRATEGIES:
            for r in R_STRATEGIES:
                in_path = RETRIEVAL_DIR / f"top20_{prer}__{r}.npy"
                out_path = TOPK_DIR / f"{prer}__{r}__{k}.npy"
                if args.skip_existing and out_path.exists():
                    continue
                if not in_path.exists():
                    print(f"  MISSING {in_path.name}, skip")
                    continue
                top20 = np.load(in_path)
                t = time.time()
                topk = rerank_one_combo(ce, chunks_text, queries, top20)
                np.save(out_path, topk)
                print(f"  {prer}__{r}: {time.time()-t:.1f}s")
        del ce


if __name__ == "__main__":
    main()
