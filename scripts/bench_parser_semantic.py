#!/usr/bin/env python3
"""Stage 2-ext: Semantic chunker 비교.

고정: loader=pymupdf (단, markdown_header 만 pymupdf4llm), embedding=gemma-embed-300m, retriever=Hybrid 3:7, top-k=5
변수: chunker

비교군:
  lc_semantic_percentile     — LC SemanticChunker, percentile breakpoint=95
  lc_semantic_stdev          — LC SemanticChunker, standard_deviation breakpoint=1.5
  lc_markdown_header         — LC MarkdownHeaderTextSplitter (pymupdf4llm 마크다운 필요)
  chonkie_semantic_500       — Chonkie SemanticChunker, chunk_size=500
  chonkie_neural             — Chonkie NeuralChunker (BERT classifier)
  llamaindex_semantic        — LlamaIndex SemanticSplitterNodeParser
  kss_recursive_500_100      — KSS 한국어 문장 분리 → 그룹화 (chunk_size 500)
  kiwi_recursive_500_100     — Kiwi 한국어 문장 분리 → 그룹화

Stage 2 winner (LC Recursive 300/50, MRR 0.6816)와 직접 비교.
"""
import argparse, json, time, sys, os, re
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import parse_pdf_pymupdf, parse_pdf_pymupdf4llm, load_ground_truth

ROOT = Path(__file__).parent.parent
PDF_DIR = ROOT / "data/pdfs"
OUT_DIR = ROOT / "results/phase2_parser_semantic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "google/embeddinggemma-300m"
RRF_K = 60
TOP_K = 5


def find_pdf(filename):
    for root, _dirs, files in os.walk(PDF_DIR):
        if filename in files:
            return Path(root) / filename
    return None


def _wrap(pages, split_fn):
    out = []
    for p in pages:
        try:
            parts = split_fn(p["text"])
        except Exception:
            parts = [p["text"]]
        for part in parts:
            part = (part or "").strip()
            if len(part) >= 30:
                out.append({"page": p["page"], "chunk_index": len(out), "text": part})
    return out


# ── chunker functions ─────────────────────────────────────────────────

def chunker_lc_semantic(threshold_type="percentile", threshold_amount=95):
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"trust_remote_code": True})
    splitter = SemanticChunker(
        embeddings=embed,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=threshold_amount,
    )
    def fn(pages):
        return _wrap(pages, splitter.split_text)
    return fn


def chunker_lc_markdown_header():
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        return_each_line=False,
    )
    def fn(pages):
        # pymupdf4llm produces markdown
        out = []
        for p in pages:
            try:
                docs = splitter.split_text(p["text"])
                for d in docs:
                    text = (d.page_content or "").strip()
                    if len(text) >= 30:
                        out.append({"page": p["page"], "chunk_index": len(out), "text": text})
            except Exception:
                if p["text"].strip():
                    out.append({"page": p["page"], "chunk_index": len(out), "text": p["text"].strip()})
        return out
    return fn


def chunker_chonkie_semantic(chunk_size=500):
    from chonkie import SemanticChunker
    c = SemanticChunker(embedding_model=EMBED_MODEL, chunk_size=chunk_size, threshold=0.5)
    def split_text(t):
        return [x.text for x in c(t)]
    def fn(pages):
        return _wrap(pages, split_text)
    return fn


def chunker_chonkie_neural():
    from chonkie import NeuralChunker
    c = NeuralChunker()
    def split_text(t):
        return [x.text for x in c(t)]
    def fn(pages):
        return _wrap(pages, split_text)
    return fn


def chunker_llamaindex_semantic():
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed = HuggingFaceEmbedding(model_name=EMBED_MODEL, trust_remote_code=True)
    splitter = SemanticSplitterNodeParser(
        embed_model=embed, buffer_size=1, breakpoint_percentile_threshold=95
    )
    from llama_index.core.schema import Document
    def split_text(t):
        nodes = splitter.get_nodes_from_documents([Document(text=t)])
        return [n.text for n in nodes]
    def fn(pages):
        return _wrap(pages, split_text)
    return fn


def chunker_kss_recursive(chunk_size=500, overlap=100):
    import kss
    def split_text(t):
        sentences = kss.split_sentences(t)
        chunks, cur = [], ""
        for s in sentences:
            if not s.strip(): continue
            if len(cur) + len(s) + 1 <= chunk_size:
                cur = f"{cur} {s}".strip()
            else:
                if cur: chunks.append(cur)
                # overlap: 끝부분 overlap 문자만큼 유지
                if overlap > 0 and len(cur) > overlap:
                    cur = cur[-overlap:] + " " + s
                else:
                    cur = s
        if cur: chunks.append(cur)
        return chunks
    def fn(pages):
        return _wrap(pages, split_text)
    return fn


def chunker_kiwi_recursive(chunk_size=500, overlap=100):
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    def split_text(t):
        sentences = [s.text for s in kiwi.split_into_sents(t)]
        chunks, cur = [], ""
        for s in sentences:
            if not s.strip(): continue
            if len(cur) + len(s) + 1 <= chunk_size:
                cur = f"{cur} {s}".strip()
            else:
                if cur: chunks.append(cur)
                if overlap > 0 and len(cur) > overlap:
                    cur = cur[-overlap:] + " " + s
                else:
                    cur = s
        if cur: chunks.append(cur)
        return chunks
    def fn(pages):
        return _wrap(pages, split_text)
    return fn


STRATEGIES = {
    "lc_semantic_percentile": ("LC SemanticChunker (percentile=95)", chunker_lc_semantic, "pymupdf"),
    "lc_semantic_stdev":      ("LC SemanticChunker (stdev=1.5)", lambda: chunker_lc_semantic("standard_deviation", 1.5), "pymupdf"),
    "lc_markdown_header":     ("LC MarkdownHeaderTextSplitter", chunker_lc_markdown_header, "pymupdf4llm"),
    "chonkie_semantic_500":   ("Chonkie SemanticChunker 500", chunker_chonkie_semantic, "pymupdf"),
    "chonkie_neural":         ("Chonkie NeuralChunker", chunker_chonkie_neural, "pymupdf"),
    "llamaindex_semantic":    ("LlamaIndex SemanticSplitter", chunker_llamaindex_semantic, "pymupdf"),
    "kss_recursive_500_100":  ("KSS + Recursive 500/100", chunker_kss_recursive, "pymupdf"),
    "kiwi_recursive_500_100": ("Kiwi + Recursive 500/100", chunker_kiwi_recursive, "pymupdf"),
}


# ── pipeline ──────────────────────────────────────────────────────────

def process_strategy(strategy_key, gt):
    label, chunker_factory, loader_name = STRATEGIES[strategy_key]
    chunker_fn = chunker_factory() if callable(chunker_factory) else chunker_factory
    chunks_all, chunk_meta = [], []
    file_set = sorted({g["target_file_name"] for g in gt})
    t0 = time.time()
    n_ok = 0
    parse_fn = parse_pdf_pymupdf4llm if loader_name == "pymupdf4llm" else parse_pdf_pymupdf
    for fname in file_set:
        path = find_pdf(fname)
        if not path: continue
        try:
            pages = parse_fn(str(path))
            chunks = chunker_fn(pages)
            for c in chunks:
                chunks_all.append(c["text"])
                chunk_meta.append({"file": fname, "page": c["page"]})
            n_ok += 1
        except Exception as e:
            print(f"  ERR {fname}: {e}")
    elapsed = time.time() - t0
    print(f"  {label}: {n_ok}/{len(file_set)} PDFs, {len(chunks_all)} chunks, {elapsed:.1f}s", flush=True)
    return {"label": label, "chunks": chunks_all, "meta": chunk_meta, "elapsed": elapsed, "n_ok": n_ok}


_model = None
_q_emb_cache = None

def get_model():
    from sentence_transformers import SentenceTransformer
    global _model
    if _model is None:
        print(f"  loading {EMBED_MODEL}...", flush=True)
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    return _model


def encode_queries(gt):
    global _q_emb_cache
    if _q_emb_cache is None:
        model = get_model()
        print(f"  encoding {len(gt)} queries...", flush=True)
        _q_emb_cache = model.encode([g["question"] for g in gt], batch_size=16,
                                     show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    return _q_emb_cache


def evaluate(chunks_text, meta, gt):
    from rank_bm25 import BM25Okapi
    from kiwipiepy import Kiwi
    model = get_model()
    kiwi = Kiwi()
    keep = ("N", "V", "X", "S", "M")
    def kt(t): return [tok.form for tok in kiwi.tokenize(t) if tok.tag.startswith(keep)]

    t0 = time.time()
    print(f"  embedding {len(chunks_text)} chunks...", flush=True)
    chunk_emb = model.encode(chunks_text, batch_size=16, show_progress_bar=True,
                             normalize_embeddings=True, convert_to_numpy=True)
    embed_t = time.time() - t0
    print(f"  chunks embedded in {embed_t:.1f}s", flush=True)

    q_emb = encode_queries(gt)
    sims = q_emb @ chunk_emb.T
    dense_ranks = np.argsort(-sims, axis=1)

    print(f"  BM25-KIWI tokenize + index...", flush=True)
    t1 = time.time()
    bm25 = BM25Okapi([kt(t) for t in chunks_text])
    q_tok = [kt(g["question"]) for g in gt]
    bm25_scores = np.array([bm25.get_scores(qt) for qt in q_tok])
    bm25_ranks = np.argsort(-bm25_scores, axis=1)
    print(f"  BM25 done in {time.time()-t1:.1f}s", flush=True)

    n_q, n_c = sims.shape
    topk = np.zeros((n_q, TOP_K), dtype=np.int64)
    for q in range(n_q):
        rank_d = np.empty(n_c, dtype=np.int64); rank_b = np.empty(n_c, dtype=np.int64)
        rank_d[dense_ranks[q]] = np.arange(n_c); rank_b[bm25_ranks[q]] = np.arange(n_c)
        scores = 0.3 / (RRF_K + rank_d + 1) + 0.7 / (RRF_K + rank_b + 1)
        top = np.argpartition(-scores, TOP_K)[:TOP_K]
        topk[q] = top[np.argsort(-scores[top])]

    hit1 = hit5 = file_hit5 = 0
    mrr = 0.0
    for i, g in enumerate(gt):
        tgt_f = g["target_file_name"]
        try:
            tgt_p = int(str(g["target_page_no"]).strip().split(",")[0].strip())
        except (ValueError, AttributeError):
            tgt_p = None
        found = None; file_found = False
        for rank, idx in enumerate(topk[i]):
            m = meta[idx]
            if m["file"] == tgt_f:
                file_found = True
                if tgt_p is None or m["page"] == tgt_p:
                    if found is None: found = rank + 1
        if found == 1: hit1 += 1
        if found and found <= 5: hit5 += 1
        if file_found: file_hit5 += 1
        if found: mrr += 1.0 / found
    n = len(gt)
    return {"MRR": mrr/n, "Hit@1": hit1/n, "Hit@5": hit5/n, "File@5": file_hit5/n, "embed_time_sec": embed_t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=["all"])
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    keys = list(STRATEGIES.keys()) if args.strategies == ["all"] else args.strategies
    gt = load_ground_truth()
    print(f"GT: {len(gt)} Q&A, target files: {len({g['target_file_name'] for g in gt})}")

    results = {}
    for k in keys:
        if k not in STRATEGIES:
            print(f"unknown: {k}"); continue
        out_path = OUT_DIR / f"{k}.json"
        if args.skip_existing and out_path.exists():
            print(f"SKIP {k}")
            results[k] = json.load(open(out_path)); continue
        print(f"\n=== {k} ===")
        try:
            s_out = process_strategy(k, gt)
            if not s_out["chunks"]: continue
            m = evaluate(s_out["chunks"], s_out["meta"], gt)
            r = {"strategy": k, "label": s_out["label"], "n_pdfs": s_out["n_ok"],
                 "n_chunks": len(s_out["chunks"]), "parse_time_sec": s_out["elapsed"], **m}
            results[k] = r
            json.dump(r, open(out_path, "w"), ensure_ascii=False, indent=2)
            print(f"  MRR={m['MRR']:.4f} Hit@1={m['Hit@1']:.3f} Hit@5={m['Hit@5']:.3f} File@5={m['File@5']:.3f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  STRATEGY ERR {k}: {e}")

    print("\n" + "=" * 100)
    print(f"{'Strategy':<32} {'Chunks':>7} {'Parse(s)':>9} {'MRR':>7} {'Hit@1':>7} {'Hit@5':>7} {'File@5':>7}")
    print("-" * 100)
    for k in sorted(results.keys(), key=lambda x: results[x].get("MRR", -1), reverse=True):
        r = results[k]
        print(f"{r['label']:<32} {r['n_chunks']:>7} {r['parse_time_sec']:>9.1f} "
              f"{r['MRR']:>7.4f} {r['Hit@1']:>7.3f} {r['Hit@5']:>7.3f} {r['File@5']:>7.3f}")


if __name__ == "__main__":
    main()
