#!/usr/bin/env python3
"""Cross-encoder Reranker 비교 — Stage 4-2.

고정 파이프라인:
  loader=pymupdf, parser=LC Recursive 300/50, embedding=embeddinggemma-300m,
  retriever=Hybrid 3:7 (dense + BM25-KIWI), top-N=20 → rerank to top-5

비교 reranker:
  - 베이스라인: rerank 없음 (top-5 그대로)
  - BAAI/bge-reranker-v2-m3 (multilingual baseline)
  - dragonkue/bge-reranker-v2-m3-ko (Korean fine-tuned)
  - jinaai/jina-reranker-v2-base-multilingual

측정: rerank 후 MRR / Hit@1 / Hit@5 / File@5

Usage:
  python scripts/bench_reranker.py --rerankers all
  python scripts/bench_reranker.py --top-n 20
"""
import argparse, json, time, sys, os, re
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import parse_pdf_pymupdf, load_ground_truth

ROOT = Path(__file__).parent.parent
PDF_DIR = ROOT / "data/pdfs"
OUT_DIR = ROOT / "results/phase4_2_reranker"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "google/embeddinggemma-300m"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RRF_K = 60
TOP_N_RETRIEVE = 20  # retrieve before rerank
TOP_K_FINAL = 5      # final after rerank

RERANKERS = {
    # Existing baseline (2024)
    "bge-reranker-v2-m3":     "BAAI/bge-reranker-v2-m3",
    "bge-reranker-v2-m3-ko":  "dragonkue/bge-reranker-v2-m3-ko",
    # 2025-2026 SOTA
    "qwen3-reranker-0.6b":    "Qwen/Qwen3-Reranker-0.6B",
    "qwen3-reranker-4b":      "Qwen/Qwen3-Reranker-4B",
    "jina-reranker-v3":       "jinaai/jina-reranker-v3",
    "mxbai-rerank-base-v2":   "mixedbread-ai/mxbai-rerank-base-v2",
    "mxbai-rerank-large-v2":  "mixedbread-ai/mxbai-rerank-large-v2",
    "bge-reranker-v2-gemma":  "BAAI/bge-reranker-v2-gemma",
    "ko-reranker":            "Dongjin-kr/ko-reranker",
    "modernReranker":         "naver/modernReranker",
    "pixie-spell-reranker":   "telepix/PIXIE-Spell-Reranker-Preview-0.6B",
    # Earlier attempts (env issues, kept for reference)
    "jina-reranker-v2":       "jinaai/jina-reranker-v2-base-multilingual",
    "gte-multilingual-reranker": "Alibaba-NLP/gte-multilingual-reranker-base",
}

# 비-cross-encoder 전략 (rerank 모델 없이)
NON_RERANKER_STRATEGIES = [
    "mmr",              # Maximum Marginal Relevance (lambda=0.5) on top-20
    "multi_rrf_2",      # RRF of bge-v2-m3-ko + qwen3-reranker-0.6b
]


# ── pipeline setup ──────────────────────────────────────────────────────

def find_pdf(filename: str):
    for root, _dirs, files in os.walk(PDF_DIR):
        if filename in files:
            return Path(root) / filename
    return None


def build_chunks(gt):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks_text, meta = [], []
    file_set = sorted({g["target_file_name"] for g in gt})
    for fname in file_set:
        path = find_pdf(fname)
        if not path: continue
        pages = parse_pdf_pymupdf(str(path))
        for p in pages:
            for part in splitter.split_text(p["text"]):
                part = part.strip()
                if len(part) >= 30:
                    chunks_text.append(part)
                    meta.append({"file": fname, "page": p["page"]})
    return chunks_text, meta


def kiwi_tokenize(text):
    if not hasattr(kiwi_tokenize, "_k"):
        from kiwipiepy import Kiwi
        kiwi_tokenize._k = Kiwi()
    keep = ("N", "V", "X", "S", "M")
    return [t.form for t in kiwi_tokenize._k.tokenize(text) if t.tag.startswith(keep)]


def hybrid_retrieve(chunks_text, gt, top_n=TOP_N_RETRIEVE):
    """dense + BM25-KIWI RRF 3:7 → top_n chunks per query."""
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    print(f"  loading dense model {EMBED_MODEL}...", flush=True)
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    print(f"  embedding {len(chunks_text)} chunks + {len(gt)} queries...", flush=True)
    chunk_emb = model.encode(chunks_text, batch_size=16, show_progress_bar=True,
                             normalize_embeddings=True, convert_to_numpy=True)
    q_emb = model.encode([g["question"] for g in gt], batch_size=16, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True)
    dense_ranks = np.argsort(-(q_emb @ chunk_emb.T), axis=1)

    print(f"  tokenizing for BM25-KIWI...", flush=True)
    chunk_tokens = [kiwi_tokenize(t) for t in chunks_text]
    bm25 = BM25Okapi(chunk_tokens)
    q_tokens = [kiwi_tokenize(g["question"]) for g in gt]
    bm25_scores = np.array([bm25.get_scores(qt) for qt in q_tokens])
    bm25_ranks = np.argsort(-bm25_scores, axis=1)

    n_q, n_c = dense_ranks.shape
    print(f"  RRF combine (3:7, top-{top_n})...", flush=True)
    out = np.zeros((n_q, top_n), dtype=np.int64)
    for q in range(n_q):
        scores = np.zeros(n_c, dtype=np.float64)
        chunk_rank_d = np.empty(n_c, dtype=np.int64)
        chunk_rank_b = np.empty(n_c, dtype=np.int64)
        chunk_rank_d[dense_ranks[q]] = np.arange(n_c)
        chunk_rank_b[bm25_ranks[q]] = np.arange(n_c)
        scores += 0.3 / (RRF_K + chunk_rank_d + 1)
        scores += 0.7 / (RRF_K + chunk_rank_b + 1)
        top = np.argpartition(-scores, top_n)[:top_n]
        out[q] = top[np.argsort(-scores[top])]
    return out


# ── reranker scoring ────────────────────────────────────────────────────

def rerank_with(model_id, chunks_text, meta, retrieved_idx, gt):
    """Apply cross-encoder reranker on top-N retrieved, return top-K reordered.

    Use LangChain HuggingFaceCrossEncoder wrapper where compatible,
    fall back to raw sentence-transformers for models needing trust_remote_code/quirks.
    """
    print(f"  loading reranker {model_id}...", flush=True)
    try:
        # LangChain wrapper first (consistent abstraction)
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        ce = HuggingFaceCrossEncoder(model_name=model_id, model_kwargs={"trust_remote_code": True})
        def score_pairs(pairs):
            return np.array(ce.score(pairs))
    except Exception as e:
        print(f"  LC wrapper failed ({e}), falling back to sentence_transformers...", flush=True)
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model_id, trust_remote_code=True)
        def score_pairs(pairs):
            return np.array(ce.predict(pairs, batch_size=16))

    n_q, top_n = retrieved_idx.shape
    out = np.zeros((n_q, TOP_K_FINAL), dtype=np.int64)
    t0 = time.time()
    pairs_all = []
    flat_origin = []
    for q in range(n_q):
        for r in range(top_n):
            idx = retrieved_idx[q, r]
            pairs_all.append([gt[q]["question"], chunks_text[idx]])
            flat_origin.append((q, idx))
    print(f"  scoring {len(pairs_all)} (q, chunk) pairs...", flush=True)
    scores = score_pairs(pairs_all)
    print(f"  reranker done in {time.time()-t0:.1f}s", flush=True)

    # gather scores per query
    per_q_scores = [[] for _ in range(n_q)]
    per_q_idx = [[] for _ in range(n_q)]
    for (q, idx), s in zip(flat_origin, scores):
        per_q_scores[q].append(s)
        per_q_idx[q].append(idx)

    for q in range(n_q):
        order = np.argsort(-np.array(per_q_scores[q]))[:TOP_K_FINAL]
        out[q] = [per_q_idx[q][i] for i in order]
    return out


# ── metrics ─────────────────────────────────────────────────────────────

def compute_metrics(topk_idx, meta, gt):
    hit1 = hit5 = file_hit5 = 0
    mrr = 0.0
    for i, g in enumerate(gt):
        tgt_f = g["target_file_name"]
        try:
            tgt_p_int = int(str(g["target_page_no"]).strip().split(",")[0].strip())
        except (ValueError, AttributeError):
            tgt_p_int = None
        found_rank = None
        file_found = False
        for rank, idx in enumerate(topk_idx[i]):
            m = meta[idx]
            if m["file"] == tgt_f:
                file_found = True
                if tgt_p_int is None or m["page"] == tgt_p_int:
                    if found_rank is None:
                        found_rank = rank + 1
        if found_rank == 1: hit1 += 1
        if found_rank and found_rank <= 5: hit5 += 1
        if file_found: file_hit5 += 1
        if found_rank: mrr += 1.0 / found_rank
    n = len(gt)
    return {"MRR": mrr/n, "Hit@1": hit1/n, "Hit@5": hit5/n, "File@5": file_hit5/n}


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerankers", nargs="+", default=["all"])
    ap.add_argument("--top-n", type=int, default=TOP_N_RETRIEVE)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    gt = load_ground_truth()
    print(f"GT: {len(gt)} Q&A")

    print("\n=== building chunks + hybrid retrieval (top-N=%d) ===" % args.top_n)
    chunks_text, meta = build_chunks(gt)
    print(f"  chunks: {len(chunks_text)}")
    cache_top = OUT_DIR / f"_retrieved_top{args.top_n}.npy"
    if cache_top.exists():
        retrieved = np.load(cache_top)
        print(f"  loaded cached retrieval: {retrieved.shape}")
    else:
        retrieved = hybrid_retrieve(chunks_text, gt, top_n=args.top_n)
        np.save(cache_top, retrieved)
        print(f"  cached retrieval saved: {retrieved.shape}")

    # baseline: no rerank, top-5 of retrieved
    print("\n=== eval: baseline (no rerank, top-5 of hybrid) ===")
    base_topk = retrieved[:, :TOP_K_FINAL]
    base = compute_metrics(base_topk, meta, gt)
    print(f"  MRR={base['MRR']:.4f} Hit@1={base['Hit@1']:.3f} Hit@5={base['Hit@5']:.3f} File@5={base['File@5']:.3f}")
    json.dump({"strategy": "no_rerank", **base}, open(OUT_DIR / "no_rerank.json", "w"), ensure_ascii=False, indent=2)

    keys = list(RERANKERS.keys()) if args.rerankers == ["all"] else args.rerankers

    results = {"no_rerank": base}
    for k in keys:
        if k not in RERANKERS: print(f"unknown: {k}"); continue
        out_path = OUT_DIR / f"{k}.json"
        if args.skip_existing and out_path.exists():
            print(f"\nSKIP {k} (cached)")
            results[k] = json.load(open(out_path))
            continue
        print(f"\n=== {k} ===")
        try:
            topk = rerank_with(RERANKERS[k], chunks_text, meta, retrieved, gt)
        except Exception as e:
            print(f"  ERR {k}: {e}")
            continue
        m = compute_metrics(topk, meta, gt)
        r = {"strategy": k, "model": RERANKERS[k], **m}
        results[k] = r
        json.dump(r, open(out_path, "w"), ensure_ascii=False, indent=2)
        print(f"  MRR={m['MRR']:.4f} Hit@1={m['Hit@1']:.3f} Hit@5={m['Hit@5']:.3f} File@5={m['File@5']:.3f}")

    print("\n" + "=" * 90)
    print(f"{'Strategy':<26} {'MRR':>7} {'Hit@1':>7} {'Hit@5':>7} {'File@5':>7}")
    print("-" * 90)
    for k in sorted(results.keys(), key=lambda x: results[x]["MRR"], reverse=True):
        r = results[k]
        print(f"{k:<26} {r['MRR']:>7.4f} {r['Hit@1']:>7.3f} {r['Hit@5']:>7.3f} {r['File@5']:>7.3f}")


if __name__ == "__main__":
    main()
