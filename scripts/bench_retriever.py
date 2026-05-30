#!/usr/bin/env python3
"""Retriever 방식 비교 — Stage 4.

고정: loader=pymupdf, parser=LC Recursive 300/50, embedding=embeddinggemma-300m, top-k=5
변수: retrieval 방식 (Dense / BM25 / Hybrid RRF)

전략:
  dense              — embeddinggemma-300m cosine (Stage 2 winner = baseline)
  bm25_whitespace    — BM25 + 공백 tokenizer
  bm25_kiwi          — BM25 + KIWI 형태소 tokenizer
  hybrid_dense_bm25_5_5    — RRF, dense:bm25_kiwi = 50:50
  hybrid_dense_bm25_7_3    — RRF, dense 가중치 우세
  hybrid_dense_bm25_3_7    — RRF, bm25 가중치 우세
  hybrid_dense_bm25ws_5_5  — RRF dense + bm25 whitespace 5:5 (kiwi 효과 측정)

RRF score(c) = Σ weight_i / (k + rank_i(c)), k=60

Usage:
  python scripts/bench_retriever.py --strategies all
"""
import argparse, json, time, sys, os, re
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from typing import List, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import parse_pdf_pymupdf, load_ground_truth

ROOT = Path(__file__).parent.parent
PDF_DIR = ROOT / "data/pdfs"
OUT_DIR = ROOT / "results/phase4_retriever_extended"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
EMBED_MODEL = "google/embeddinggemma-300m"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RRF_K = 60


# ── chunk 만들기 (loader=pymupdf, parser=LC Recursive 300/50) ────────────

def find_pdf(filename: str) -> Path:
    for root, _dirs, files in os.walk(PDF_DIR):
        if filename in files:
            return Path(root) / filename
    return None


def build_chunks(gt: list):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks_text, chunk_meta = [], []
    file_set = sorted({g["target_file_name"] for g in gt})
    t0 = time.time()
    for fname in file_set:
        path = find_pdf(fname)
        if not path:
            continue
        pages = parse_pdf_pymupdf(str(path))
        for p in pages:
            for part in splitter.split_text(p["text"]):
                part = part.strip()
                if len(part) >= 30:
                    chunks_text.append(part)
                    chunk_meta.append({"file": fname, "page": p["page"]})
    print(f"  built {len(chunks_text)} chunks in {time.time()-t0:.1f}s", flush=True)
    return chunks_text, chunk_meta


# ── tokenizers ──────────────────────────────────────────────────────────

_kiwi = None

def kiwi_tokenize(text: str) -> List[str]:
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    # 명사·동사·형용사·어근 위주 (조사·어미 제거)
    keep_prefix = ("N", "V", "X", "S", "M")  # noun/verb/adjective/root/adverb/symbol/number
    return [t.form for t in _kiwi.tokenize(text) if t.tag.startswith(keep_prefix)]


def whitespace_tokenize(text: str) -> List[str]:
    # 한글/영문/숫자만 유지, 공백 split
    cleaned = re.sub(r"[^\w가-힣\s]", " ", text)
    return [t for t in cleaned.lower().split() if t]


# ── retrievers ──────────────────────────────────────────────────────────

def search_dense(chunks_text, gt):
    from sentence_transformers import SentenceTransformer
    print(f"  loading {EMBED_MODEL}...", flush=True)
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    print(f"  embedding {len(chunks_text)} chunks...", flush=True)
    t = time.time()
    chunk_emb = model.encode(chunks_text, batch_size=16, show_progress_bar=True,
                             normalize_embeddings=True, convert_to_numpy=True)
    queries = [g["question"] for g in gt]
    print(f"  embedding {len(queries)} queries...", flush=True)
    q_emb = model.encode(queries, batch_size=16, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True)
    sims = q_emb @ chunk_emb.T
    print(f"  dense done in {time.time()-t:.1f}s", flush=True)
    # full ranking (sorted desc) — for RRF we need ranks of all chunks per query
    ranks = np.argsort(-sims, axis=1)
    return ranks, sims


def search_bm25(chunks_text, gt, tokenizer):
    from rank_bm25 import BM25Okapi
    t = time.time()
    print(f"  tokenizing {len(chunks_text)} chunks...", flush=True)
    chunk_tokens = [tokenizer(t) for t in chunks_text]
    print(f"  building BM25 index...", flush=True)
    bm25 = BM25Okapi(chunk_tokens)
    print(f"  scoring {len(gt)} queries...", flush=True)
    q_tokens = [tokenizer(g["question"]) for g in gt]
    scores = np.array([bm25.get_scores(qt) for qt in q_tokens])  # (n_q, n_chunks)
    print(f"  bm25 done in {time.time()-t:.1f}s", flush=True)
    ranks = np.argsort(-scores, axis=1)
    return ranks, scores


def rrf_combine(rank_lists: List[np.ndarray], weights: List[float], n_chunks: int, top_k=TOP_K):
    """rank_lists: list of (n_q, n_chunks) integer arrays, each row = chunks sorted by rank.
    Returns (n_q, top_k) array of chunk indices."""
    n_q = rank_lists[0].shape[0]
    out = np.zeros((n_q, top_k), dtype=np.int64)
    for q in range(n_q):
        scores = np.zeros(n_chunks, dtype=np.float64)
        for ranks, w in zip(rank_lists, weights):
            # ranks[q] is order of chunk indices; we need rank of each chunk
            chunk_rank = np.empty(n_chunks, dtype=np.int64)
            chunk_rank[ranks[q]] = np.arange(n_chunks)
            scores += w / (RRF_K + chunk_rank + 1)  # +1: rank starts at 1
        top = np.argpartition(-scores, top_k)[:top_k]
        top = top[np.argsort(-scores[top])]
        out[q] = top
    return out


# ── metrics ─────────────────────────────────────────────────────────────

def compute_metrics(topk_idx, meta, gt):
    hit1 = hit5 = file_hit5 = 0
    mrr = 0.0
    for i, g in enumerate(gt):
        tgt_f = g["target_file_name"]
        tgt_p = str(g["target_page_no"]).strip()
        try:
            tgt_p_int = int(tgt_p.split(",")[0].strip())
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
    ap.add_argument("--strategies", nargs="+", default=["all"])
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    gt = load_ground_truth()
    print(f"GT: {len(gt)} Q&A, target files: {len({g['target_file_name'] for g in gt})}")

    print("\n=== building chunks (pymupdf + LC Recursive 300/50) ===")
    chunks_text, meta = build_chunks(gt)
    n_chunks = len(chunks_text)

    all_strategies = [
        "dense",
        "bm25_whitespace",
        "bm25_kiwi",
        "hybrid_5_5",
        "hybrid_7_3",
        "hybrid_3_7",
        "hybrid_ws_5_5",
    ]
    strategies = all_strategies if args.strategies == ["all"] else args.strategies

    # precompute base retrievers as needed
    cache = {}
    def need(name):
        return any(s == name or s.startswith("hybrid") for s in strategies)

    if "dense" in strategies or any(s.startswith("hybrid") for s in strategies):
        print("\n=== Dense retrieval ===")
        cache["dense_ranks"], _ = search_dense(chunks_text, gt)
    if "bm25_kiwi" in strategies or any(s in strategies for s in ["hybrid_5_5", "hybrid_7_3", "hybrid_3_7"]):
        print("\n=== BM25 + KIWI ===")
        cache["bm25_kiwi_ranks"], _ = search_bm25(chunks_text, gt, kiwi_tokenize)
    if "bm25_whitespace" in strategies or "hybrid_ws_5_5" in strategies:
        print("\n=== BM25 whitespace ===")
        cache["bm25_ws_ranks"], _ = search_bm25(chunks_text, gt, whitespace_tokenize)

    results = {}
    for s in strategies:
        out_path = OUT_DIR / f"{s}.json"
        if args.skip_existing and out_path.exists():
            print(f"\nSKIP {s} (cached)")
            results[s] = json.load(open(out_path))
            continue
        print(f"\n=== eval: {s} ===")
        if s == "dense":
            topk = cache["dense_ranks"][:, :TOP_K]
        elif s == "bm25_kiwi":
            topk = cache["bm25_kiwi_ranks"][:, :TOP_K]
        elif s == "bm25_whitespace":
            topk = cache["bm25_ws_ranks"][:, :TOP_K]
        elif s == "hybrid_5_5":
            topk = rrf_combine([cache["dense_ranks"], cache["bm25_kiwi_ranks"]], [0.5, 0.5], n_chunks)
        elif s == "hybrid_7_3":
            topk = rrf_combine([cache["dense_ranks"], cache["bm25_kiwi_ranks"]], [0.7, 0.3], n_chunks)
        elif s == "hybrid_3_7":
            topk = rrf_combine([cache["dense_ranks"], cache["bm25_kiwi_ranks"]], [0.3, 0.7], n_chunks)
        elif s == "hybrid_ws_5_5":
            topk = rrf_combine([cache["dense_ranks"], cache["bm25_ws_ranks"]], [0.5, 0.5], n_chunks)
        else:
            print(f"unknown: {s}"); continue
        m = compute_metrics(topk, meta, gt)
        r = {"strategy": s, "n_chunks": n_chunks, **m}
        results[s] = r
        json.dump(r, open(out_path, "w"), ensure_ascii=False, indent=2)
        print(f"  MRR={m['MRR']:.4f} Hit@1={m['Hit@1']:.3f} Hit@5={m['Hit@5']:.3f} File@5={m['File@5']:.3f}")

    print("\n" + "=" * 90)
    print(f"{'Strategy':<24} {'MRR':>7} {'Hit@1':>7} {'Hit@5':>7} {'File@5':>7}")
    print("-" * 90)
    for s in sorted(strategies, key=lambda x: results.get(x, {}).get("MRR", -1), reverse=True):
        r = results.get(s)
        if not r: continue
        print(f"{s:<24} {r['MRR']:>7.4f} {r['Hit@1']:>7.3f} {r['Hit@5']:>7.3f} {r['File@5']:>7.3f}")


if __name__ == "__main__":
    main()
