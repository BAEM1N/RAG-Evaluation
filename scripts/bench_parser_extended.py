#!/usr/bin/env python3
"""Parser 비교 실험 — Stage 2.

고정: loader=pymupdf, embedding=google/embeddinggemma-300m, top-k=5
변수: chunker 라이브러리 + 파라미터

비교군:
  LC RecursiveCharacterTextSplitter @ {300/50, 500/100, 800/150, 1000/200, 1500/200}
  LC CharacterTextSplitter @ 500/100
  LC TokenTextSplitter @ 256/50 tokens (tiktoken)
  Chonkie TokenChunker @ 256/50 tokens
  Chonkie SentenceChunker (~500 chars)
  Chonkie RecursiveChunker @ 500 chars
  Chonkie FastChunker @ 500 chars
  LlamaIndex SentenceSplitter @ 500/100

Usage:
  python scripts/bench_parser_extended.py --strategies all
  python scripts/bench_parser_extended.py --strategies lc_recursive_500_100 chonkie_sentence
  python scripts/bench_parser_extended.py --skip-existing
"""
import argparse, json, time, sys, os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from typing import List, Dict, Callable

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import parse_pdf_pymupdf, load_ground_truth

ROOT = Path(__file__).parent.parent
PDF_DIR = ROOT / "data/pdfs"
OUT_DIR = ROOT / "results/phase2_parser_extended"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
EMBED_MODEL = "google/embeddinggemma-300m"


# ── chunker 정의 ────────────────────────────────────────────────────────
# 입력: pages = [{"page": int, "text": str}, ...]
# 출력: chunks = [{"page": int, "chunk_index": int, "text": str}, ...]

def _wrap_page_chunks(pages: List[Dict], split_fn: Callable[[str], List[str]]) -> List[Dict]:
    """페이지별 split_fn 적용 후 chunk dict로 변환."""
    out = []
    for p in pages:
        try:
            parts = split_fn(p["text"])
        except Exception:
            parts = [p["text"]]
        for part in parts:
            part = part.strip()
            if len(part) >= 30:
                out.append({"page": p["page"], "chunk_index": len(out), "text": part})
    return out


def chunker_lc_recursive(size: int, overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    s = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    def fn(pages):
        return _wrap_page_chunks(pages, s.split_text)
    return fn


def chunker_lc_character(size: int, overlap: int):
    from langchain_text_splitters import CharacterTextSplitter
    s = CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separator="\n")
    def fn(pages):
        return _wrap_page_chunks(pages, s.split_text)
    return fn


def chunker_lc_token(tokens: int, overlap: int):
    from langchain_text_splitters import TokenTextSplitter
    s = TokenTextSplitter(chunk_size=tokens, chunk_overlap=overlap, encoding_name="cl100k_base")
    def fn(pages):
        return _wrap_page_chunks(pages, s.split_text)
    return fn


def chunker_chonkie_token(tokens: int, overlap: int):
    from chonkie import TokenChunker
    c = TokenChunker(tokenizer="gpt2", chunk_size=tokens, chunk_overlap=overlap)
    def split_text(t: str):
        return [x.text for x in c(t)]
    def fn(pages):
        return _wrap_page_chunks(pages, split_text)
    return fn


def chunker_chonkie_sentence(chunk_size: int, overlap: int):
    from chonkie import SentenceChunker
    c = SentenceChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    def split_text(t: str):
        return [x.text for x in c(t)]
    def fn(pages):
        return _wrap_page_chunks(pages, split_text)
    return fn


def chunker_chonkie_recursive(chunk_size: int):
    from chonkie import RecursiveChunker
    c = RecursiveChunker(chunk_size=chunk_size)
    def split_text(t: str):
        return [x.text for x in c(t)]
    def fn(pages):
        return _wrap_page_chunks(pages, split_text)
    return fn


def chunker_chonkie_fast(chunk_size: int):
    from chonkie import FastChunker
    c = FastChunker(chunk_size=chunk_size)
    def split_text(t: str):
        return [x.text for x in c(t)]
    def fn(pages):
        return _wrap_page_chunks(pages, split_text)
    return fn


def chunker_llamaindex_sentence(chunk_size: int, overlap: int):
    from llama_index.core.node_parser import SentenceSplitter
    s = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    def split_text(t: str):
        return s.split_text(t)
    def fn(pages):
        return _wrap_page_chunks(pages, split_text)
    return fn


STRATEGIES = {
    "lc_recursive_300_50":   ("LC Recursive 300/50",   chunker_lc_recursive(300, 50)),
    "lc_recursive_500_100":  ("LC Recursive 500/100",  chunker_lc_recursive(500, 100)),
    "lc_recursive_800_150":  ("LC Recursive 800/150",  chunker_lc_recursive(800, 150)),
    "lc_recursive_1000_200": ("LC Recursive 1000/200", chunker_lc_recursive(1000, 200)),
    "lc_recursive_1500_200": ("LC Recursive 1500/200", chunker_lc_recursive(1500, 200)),
    "lc_character_500_100":  ("LC Character 500/100",  chunker_lc_character(500, 100)),
    "lc_token_256_50":       ("LC Token 256/50",       chunker_lc_token(256, 50)),
    "chonkie_token_256_50":  ("Chonkie Token 256/50",  chunker_chonkie_token(256, 50)),
    "chonkie_sentence_500":  ("Chonkie Sentence 500",  chunker_chonkie_sentence(500, 100)),
    "chonkie_recursive_500": ("Chonkie Recursive 500", chunker_chonkie_recursive(500)),
    "chonkie_fast_500":      ("Chonkie Fast 500",      chunker_chonkie_fast(500)),
    "llamaindex_sentence_500_100": ("LlamaIndex Sentence 500/100", chunker_llamaindex_sentence(500, 100)),
    # ── 300/50 공정 비교 추가 ─────────────────────────────────────────
    "lc_character_300_50":   ("LC Character 300/50",   chunker_lc_character(300, 50)),
    "chonkie_sentence_300":  ("Chonkie Sentence 300",  chunker_chonkie_sentence(300, 50)),
    "chonkie_recursive_300": ("Chonkie Recursive 300", chunker_chonkie_recursive(300)),
    "chonkie_fast_300":      ("Chonkie Fast 300",      chunker_chonkie_fast(300)),
    "llamaindex_sentence_300_50": ("LlamaIndex Sentence 300/50", chunker_llamaindex_sentence(300, 50)),
    # ── size grid 전체 확장 (각 chunker × 800/150, 1000/200, 1500/200) ──
    "lc_character_800_150":     ("LC Character 800/150",     chunker_lc_character(800, 150)),
    "lc_character_1000_200":    ("LC Character 1000/200",    chunker_lc_character(1000, 200)),
    "lc_character_1500_200":    ("LC Character 1500/200",    chunker_lc_character(1500, 200)),
    "chonkie_sentence_800":     ("Chonkie Sentence 800",     chunker_chonkie_sentence(800, 150)),
    "chonkie_sentence_1000":    ("Chonkie Sentence 1000",    chunker_chonkie_sentence(1000, 200)),
    "chonkie_sentence_1500":    ("Chonkie Sentence 1500",    chunker_chonkie_sentence(1500, 200)),
    "chonkie_recursive_800":    ("Chonkie Recursive 800",    chunker_chonkie_recursive(800)),
    "chonkie_recursive_1000":   ("Chonkie Recursive 1000",   chunker_chonkie_recursive(1000)),
    "chonkie_recursive_1500":   ("Chonkie Recursive 1500",   chunker_chonkie_recursive(1500)),
    "chonkie_fast_800":         ("Chonkie Fast 800",         chunker_chonkie_fast(800)),
    "chonkie_fast_1000":        ("Chonkie Fast 1000",        chunker_chonkie_fast(1000)),
    "chonkie_fast_1500":        ("Chonkie Fast 1500",        chunker_chonkie_fast(1500)),
    "llamaindex_sentence_800_150":  ("LlamaIndex Sentence 800/150",  chunker_llamaindex_sentence(800, 150)),
    "llamaindex_sentence_1000_200": ("LlamaIndex Sentence 1000/200", chunker_llamaindex_sentence(1000, 200)),
    "llamaindex_sentence_1500_200": ("LlamaIndex Sentence 1500/200", chunker_llamaindex_sentence(1500, 200)),
}


# ── 공통: PDF → chunk ──────────────────────────────────────────────────

def find_pdf(filename: str) -> Path:
    for root, _dirs, files in os.walk(PDF_DIR):
        if filename in files:
            return Path(root) / filename
    return None


def process_strategy(strategy_key: str, gt: list) -> dict:
    label, chunker_fn = STRATEGIES[strategy_key]
    chunks_all, chunk_meta = [], []
    file_set = sorted({g["target_file_name"] for g in gt})
    t0 = time.time()
    n_ok = 0
    for fname in file_set:
        path = find_pdf(fname)
        if not path:
            continue
        try:
            pages = parse_pdf_pymupdf(str(path))
            chunks = chunker_fn(pages)
            for c in chunks:
                chunks_all.append(c["text"])
                chunk_meta.append({"file": fname, "page": c["page"], "chunk_index": c["chunk_index"]})
            n_ok += 1
        except Exception as e:
            print(f"  ERR {fname}: {e}")
    elapsed = time.time() - t0
    print(f"  {label}: {n_ok}/{len(file_set)} PDFs, {len(chunks_all)} chunks, {elapsed:.1f}s", flush=True)
    return {"label": label, "chunks": chunks_all, "meta": chunk_meta, "elapsed": elapsed, "n_ok": n_ok}


# ── retrieval 평가 ─────────────────────────────────────────────────────

_model = None
_q_emb_cache = None


def get_model():
    from sentence_transformers import SentenceTransformer
    global _model
    if _model is None:
        print(f"  loading {EMBED_MODEL}...", flush=True)
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
        print("  model loaded.", flush=True)
    return _model


def encode_queries(gt):
    import numpy as np
    global _q_emb_cache
    if _q_emb_cache is None:
        model = get_model()
        queries = [g["question"] for g in gt]
        print(f"  encoding {len(queries)} queries (cached for all strategies)...", flush=True)
        t = time.time()
        q = model.encode(queries, batch_size=16, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
        _q_emb_cache = np.asarray(q, dtype="float32")
        print(f"  queries encoded in {time.time()-t:.1f}s", flush=True)
    return _q_emb_cache


def evaluate(chunks_text, meta, gt) -> dict:
    import numpy as np
    model = get_model()

    t0 = time.time()
    print(f"  embedding {len(chunks_text)} chunks...", flush=True)
    chunk_emb = model.encode(chunks_text, batch_size=16, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    chunk_emb = np.asarray(chunk_emb, dtype="float32")
    embed_t = time.time() - t0
    print(f"  chunks embedded in {embed_t:.1f}s, shape={chunk_emb.shape}", flush=True)

    q_emb = encode_queries(gt)

    print(f"  computing top-{TOP_K}...", flush=True)
    t1 = time.time()
    sims = q_emb @ chunk_emb.T
    indices = np.argpartition(-sims, min(TOP_K, sims.shape[1] - 1), axis=1)[:, :TOP_K]
    for i in range(indices.shape[0]):
        order = np.argsort(-sims[i, indices[i]])
        indices[i] = indices[i][order]
    print(f"  search done in {time.time()-t1:.2f}s", flush=True)

    hit1 = hit5 = file_hit5 = 0
    mrr_total = 0.0
    for i, g in enumerate(gt):
        target_file = g["target_file_name"]
        target_page = str(g["target_page_no"]).strip()
        try:
            target_page_int = int(target_page.split(",")[0].strip())
        except (ValueError, AttributeError):
            target_page_int = None

        topk = indices[i]
        found_rank = None
        file_found = False
        for rank, idx in enumerate(topk):
            m = meta[idx]
            if m["file"] == target_file:
                file_found = True
                if target_page_int is None or m["page"] == target_page_int:
                    if found_rank is None:
                        found_rank = rank + 1
        if found_rank == 1:
            hit1 += 1
        if found_rank and found_rank <= 5:
            hit5 += 1
        if file_found:
            file_hit5 += 1
        if found_rank:
            mrr_total += 1.0 / found_rank

    n = len(gt)
    return {
        "MRR": mrr_total / n,
        "Hit@1": hit1 / n,
        "Hit@5": hit5 / n,
        "File@5": file_hit5 / n,
        "embed_time_sec": embed_t,
    }


# ── main ────────────────────────────────────────────────────────────────

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
            print(f"unknown strategy: {k}")
            continue
        out_path = OUT_DIR / f"{k}.json"
        if args.skip_existing and out_path.exists():
            print(f"\nSKIP {k} (cached)")
            results[k] = json.load(open(out_path))
            continue
        print(f"\n=== {k} ===")
        s_out = process_strategy(k, gt)
        if not s_out["chunks"]:
            continue
        m = evaluate(s_out["chunks"], s_out["meta"], gt)
        r = {
            "strategy": k, "label": s_out["label"],
            "n_pdfs": s_out["n_ok"], "n_chunks": len(s_out["chunks"]),
            "parse_time_sec": s_out["elapsed"], "embed_time_sec": m["embed_time_sec"],
            "MRR": m["MRR"], "Hit@1": m["Hit@1"], "Hit@5": m["Hit@5"], "File@5": m["File@5"],
        }
        results[k] = r
        json.dump(r, open(out_path, "w"), ensure_ascii=False, indent=2)
        print(f"  MRR={m['MRR']:.4f} Hit@1={m['Hit@1']:.3f} Hit@5={m['Hit@5']:.3f} File@5={m['File@5']:.3f}")

    print("\n" + "=" * 100)
    print(f"{'Strategy':<32} {'Chunks':>7} {'Parse(s)':>9} {'MRR':>7} {'Hit@1':>7} {'Hit@5':>7} {'File@5':>7}")
    print("-" * 100)
    sorted_keys = sorted(keys, key=lambda x: results.get(x, {}).get("MRR", -1), reverse=True)
    for k in sorted_keys:
        r = results.get(k)
        if not r: continue
        print(f"{r['label']:<32} {r['n_chunks']:>7} {r['parse_time_sec']:>9.1f} {r['MRR']:>7.4f} {r['Hit@1']:>7.3f} {r['Hit@5']:>7.3f} {r['File@5']:>7.3f}")


if __name__ == "__main__":
    main()
