#!/usr/bin/env python3
"""Step 1: pre-compute retrieval matrix (8 PreR × 6 R = 48 unique top-20).

각 (PreR, R) 조합에 대해 query별 top-20 chunk indices를 .npy로 저장.
이 후 PostR 단계에서 재사용 → reranker 입력만 변경.

캐시 위치: results/cartesian/retrieval/top20_<prer>__<r>.npy  (shape: 300×20)
공통 chunks/queries/meta는 _chunks_cache.json (Stage 4-2와 동일)에서 로드.

Usage:
  python scripts/cartesian/retrieval_matrix.py [--skip-existing]
"""
import argparse, json, os, sys, re, time
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cartesian.config import (
    PRER_STRATEGIES, R_STRATEGIES, EMBED_MODEL, RRF_K, TOP_N_RERANK,
)

CHUNK_CACHE = ROOT / "results/phase4_2_reranker/_chunks_cache.json"
PRER_LLM_CACHE = ROOT / "results/phase4_1_pre_retriever/_llm_cache_gpt54"
OUT_DIR = ROOT / "results/cartesian/retrieval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def kiwi_tokenize(text):
    if not hasattr(kiwi_tokenize, "_k"):
        from kiwipiepy import Kiwi
        kiwi_tokenize._k = Kiwi()
    keep = ("N", "V", "X", "S", "M")
    return [t.form for t in kiwi_tokenize._k.tokenize(text) if t.tag.startswith(keep)]


def whitespace_tokenize(text):
    cleaned = re.sub(r"[^\w가-힣\s]", " ", text)
    return [t for t in cleaned.lower().split() if t]


def parse_lines(text, n, fallback):
    if not text: return [fallback]*n
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in lines]
    lines = [ln for ln in lines if len(ln) > 2][:n]
    while len(lines) < n: lines.append(fallback)
    return lines


def parse_keywords(text):
    if not text: return []
    raw = text.replace("\n", ",").split(",")
    kws = [w.strip() for w in raw if w.strip()]
    kws = [re.sub(r"^[\-\*\d\.\)\s]+", "", w).strip() for w in kws]
    return [w for w in kws if 1 < len(w) < 30][:5]


def load_cache():
    d = json.load(open(CHUNK_CACHE, encoding="utf-8"))
    return d["chunks_text"], d["meta"], d["queries"], d["gt"]


def load_prer_outputs():
    """Load all Stage 4-1 cached LLM outputs (300 queries × 7 prompt types)."""
    out = {}
    for f in PRER_LLM_CACHE.glob("*.json"):
        # filename: hyde_q0000.json
        name = f.stem
        out[name] = json.load(open(f))["output"]
    return out


def get_prer_inputs(prer, qid, q, llm_cache):
    """Return list of (dense_q, bm25_q) tuples for this PreR strategy."""
    L = llm_cache
    def load(key):
        return L.get(f"{key}_{qid}")
    if prer == "baseline":
        return [(q, q)]
    if prer == "hyde":
        ans = load("hyde") or q
        return [(ans, q)]
    if prer == "hyde_rrf":
        ans = load("hyde") or q
        return [(q, q), (ans, q)]
    if prer == "query2doc":
        doc = load("query2doc") or q
        return [(f"{q} {doc}", f"{q} {doc}")]
    if prer == "multi_query_para":
        out = load("multiq_para")
        vs = parse_lines(out, 3, q) if out else [q]*3
        return [(q, q)] + [(v, v) for v in vs]
    if prer == "decompose":
        out = load("decompose")
        subs = parse_lines(out, 3, q) if out else [q]*3
        return [(q, q)] + [(s, s) for s in subs]
    if prer == "query_expansion":
        out = load("expand") or ""
        kws = parse_keywords(out)
        expanded = f"{q} {' '.join(kws)}" if kws else q
        return [(expanded, expanded)]
    if prer == "query_rewrite":
        rew = (load("rewrite") or q).split("\n")[0].strip() or q
        return [(rew, rew)]
    raise ValueError(prer)


def precompute_chunk_indices(chunks_text):
    """Embed chunks + build BM25 indices once."""
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    print(f"  loading {EMBED_MODEL}...", flush=True)
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    print(f"  embedding {len(chunks_text)} chunks...", flush=True)
    chunk_emb = model.encode(chunks_text, batch_size=16, show_progress_bar=True,
                             normalize_embeddings=True, convert_to_numpy=True)
    print(f"  BM25-KIWI tokenize...", flush=True)
    bm25_kiwi = BM25Okapi([kiwi_tokenize(t) for t in chunks_text])
    print(f"  BM25-whitespace tokenize...", flush=True)
    bm25_ws = BM25Okapi([whitespace_tokenize(t) for t in chunks_text])
    return model, chunk_emb, bm25_kiwi, bm25_ws


def hybrid_scores(model, chunk_emb, bm25_kiwi, bm25_ws, r, dq, bq):
    """Return (n_chunks,) score array for (PreR-output, R-strategy)."""
    q_emb = model.encode([dq], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    sims = (q_emb @ chunk_emb.T)[0]
    dense_rank = np.argsort(-sims)
    n_c = len(sims)
    rank_d = np.empty(n_c, dtype=np.int64); rank_d[dense_rank] = np.arange(n_c)
    if r == "dense":
        return -rank_d.astype(np.float64)
    # bm25 routing
    if r == "bm25_whitespace" or r == "hybrid_ws_5_5":
        bm25 = bm25_ws
        bq_toks = whitespace_tokenize(bq)
    else:
        bm25 = bm25_kiwi
        bq_toks = kiwi_tokenize(bq)
    bm25_scores = bm25.get_scores(bq_toks)
    bm25_rank = np.argsort(-bm25_scores)
    rank_b = np.empty(n_c, dtype=np.int64); rank_b[bm25_rank] = np.arange(n_c)
    if r == "bm25_kiwi" or r == "bm25_whitespace":
        return -rank_b.astype(np.float64)
    if r == "hybrid_7_3": w_d, w_b = 0.7, 0.3
    elif r == "hybrid_5_5": w_d, w_b = 0.5, 0.5
    elif r == "hybrid_3_7": w_d, w_b = 0.3, 0.7
    elif r == "hybrid_ws_5_5": w_d, w_b = 0.5, 0.5
    else: raise ValueError(r)
    return w_d / (RRF_K + rank_d + 1) + w_b / (RRF_K + rank_b + 1)


def fuse_and_topk(score_list, n_chunks, top_n):
    fused = np.zeros(n_chunks, dtype=np.float64)
    for s in score_list: fused += s
    top = np.argpartition(-fused, top_n)[:top_n]
    return top[np.argsort(-fused[top])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    chunks_text, meta, queries, gt = load_cache()
    n_chunks = len(chunks_text)
    print(f"chunks: {n_chunks}, queries: {len(queries)}")

    llm_cache = load_prer_outputs()
    print(f"PreR LLM cache: {len(llm_cache)} entries")

    model, chunk_emb, bm25_kiwi, bm25_ws = precompute_chunk_indices(chunks_text)

    total = len(PRER_STRATEGIES) * len(R_STRATEGIES)
    done = 0
    t_total = time.time()
    for prer in PRER_STRATEGIES:
        for r in R_STRATEGIES:
            done += 1
            out_path = OUT_DIR / f"top20_{prer}__{r}.npy"
            if args.skip_existing and out_path.exists():
                print(f"[{done}/{total}] SKIP {prer}__{r}")
                continue
            t = time.time()
            top20 = np.zeros((len(queries), TOP_N_RERANK), dtype=np.int64)
            for i, q in enumerate(queries):
                qid = f"q{i:04d}"
                inputs = get_prer_inputs(prer, qid, q, llm_cache)
                score_list = [hybrid_scores(model, chunk_emb, bm25_kiwi, bm25_ws, r, dq, bq)
                              for dq, bq in inputs]
                top20[i] = fuse_and_topk(score_list, n_chunks, TOP_N_RERANK)
            np.save(out_path, top20)
            print(f"[{done}/{total}] {prer}__{r}: {time.time()-t:.1f}s")
    print(f"\nTotal: {time.time()-t_total:.1f}s, saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
