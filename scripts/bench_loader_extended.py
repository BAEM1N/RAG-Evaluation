#!/usr/bin/env python3
"""Loader 7종 비교 실험.

7 loader × 58 PDF → text 추출 → 동일 chunker(1000/200) → 동일 embedding(sentence-transformers koe5)
→ FAISS top-k=5 retrieval → MRR / Hit@1 / Hit@5 / File@5 측정.

Usage:
  python scripts/bench_loader_extended.py --loaders all
  python scripts/bench_loader_extended.py --loaders pdfplumber pdfminer
  python scripts/bench_loader_extended.py --skip-existing  # 이미 처리한 loader 스킵
"""
import argparse, json, time, sys, os, re
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import (
    parse_pdf_pypdf, parse_pdf_pymupdf4llm, parse_pdf_pymupdf,
    chunk_pages, load_ground_truth,
)

ROOT = Path(__file__).parent.parent
PDF_DIR = ROOT / "data/pdfs"
OUT_DIR = ROOT / "results/phase1_parser_extended"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
EMBED_MODEL = "google/embeddinggemma-300m"  # phase 5 retrieval default


# ── 새 loader 4종 ────────────────────────────────────────────────────────

def parse_pdf_pdfplumber(pdf_path: str) -> List[Dict]:
    import pdfplumber
    pages = []
    with pdfplumber.open(pdf_path) as doc:
        for i, page in enumerate(doc.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


def parse_pdf_pdfminer(pdf_path: str) -> List[Dict]:
    from pdfminer.high_level import extract_text
    from pdfminer.pdfpage import PDFPage
    pages = []
    for i, page_layout in enumerate(PDFPage.get_pages(open(pdf_path, "rb"))):
        text = extract_text(pdf_path, page_numbers=[i]).strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def parse_pdf_docling(pdf_path: str) -> List[Dict]:
    from docling.document_converter import DocumentConverter
    if not hasattr(parse_pdf_docling, "_conv"):
        parse_pdf_docling._conv = DocumentConverter()
    conv = parse_pdf_docling._conv
    res = conv.convert(pdf_path)
    doc = res.document
    # page-level export
    pages = []
    for page_no in sorted(doc.pages.keys()):
        try:
            md = doc.export_to_markdown(page_no=page_no).strip()
        except Exception:
            md = ""
        if md:
            pages.append({"page": page_no, "text": md})
    return pages


def parse_pdf_opendataloader(pdf_path: str) -> List[Dict]:
    from opendataloader_pdf import convert
    out = Path("/tmp/odl_loader_bench")
    out.mkdir(exist_ok=True)
    SEP = "===ODL-PAGE-BREAK==="
    convert(
        input_path=pdf_path, output_dir=str(out), format="markdown",
        markdown_page_separator=f"\n\n{SEP}\n\n", quiet=True,
    )
    md_path = out / (Path(pdf_path).stem + ".md")
    if not md_path.exists():
        md_path = next(out.rglob(f"{Path(pdf_path).stem}.md"), None)
    if not md_path:
        return []
    text = md_path.read_text()
    parts = [p.strip() for p in text.split(SEP)]
    parts = [p for p in parts if p]
    return [{"page": i + 1, "text": p} for i, p in enumerate(parts)]


LOADERS = {
    "pypdf": parse_pdf_pypdf,
    "pymupdf": parse_pdf_pymupdf,
    "pymupdf4llm": parse_pdf_pymupdf4llm,
    "pdfplumber": parse_pdf_pdfplumber,
    "pdfminer": parse_pdf_pdfminer,
    "docling": parse_pdf_docling,
    "opendataloader": parse_pdf_opendataloader,
}


# ── PDF 처리: loader별 chunk 생성 ────────────────────────────────────────

def find_pdf(filename: str) -> Path:
    # os.walk avoids glob bracket issues with Korean filenames like [민사]
    for root, _dirs, files in os.walk(PDF_DIR):
        if filename in files:
            return Path(root) / filename
    return None


def process_loader(loader_name: str, gt: list) -> dict:
    parse_fn = LOADERS[loader_name]
    chunks_all = []
    chunk_meta = []  # (file_name, page, chunk_idx) for retrieval matching
    file_set = sorted({g["target_file_name"] for g in gt})
    t0 = time.time()
    n_ok = 0
    for fname in file_set:
        path = find_pdf(fname)
        if not path:
            continue
        try:
            pages = parse_fn(str(path))
            chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
            for c in chunks:
                chunks_all.append(c["text"])
                chunk_meta.append({"file": fname, "page": c["page"], "chunk_index": c["chunk_index"]})
            n_ok += 1
        except Exception as e:
            print(f"  ERR {fname}: {e}")
    elapsed = time.time() - t0
    print(f"  {loader_name}: {n_ok}/{len(file_set)} PDFs, {len(chunks_all)} chunks, {elapsed:.1f}s")
    return {"chunks": chunks_all, "meta": chunk_meta, "elapsed": elapsed, "n_ok": n_ok}


# ── retrieval 평가 ──────────────────────────────────────────────────────

def evaluate(chunks_text: list, meta: list, gt: list) -> dict:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    if not hasattr(evaluate, "_model"):
        print(f"  loading {EMBED_MODEL}...", flush=True)
        evaluate._model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
        print("  model loaded.", flush=True)
    model = evaluate._model

    # embed chunks in smaller batches to avoid mac mp issues
    t0 = time.time()
    print(f"  embedding {len(chunks_text)} chunks...", flush=True)
    chunk_emb = model.encode(chunks_text, batch_size=16, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    chunk_emb = np.asarray(chunk_emb, dtype="float32")
    embed_t = time.time() - t0
    print(f"  chunks embedded in {embed_t:.1f}s, shape={chunk_emb.shape}", flush=True)

    # embed queries
    queries = [g["question"] for g in gt]
    print(f"  encoding {len(queries)} queries...", flush=True)
    t1 = time.time()
    q_emb = model.encode(queries, batch_size=16, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    print(f"  queries embedded in {time.time()-t1:.1f}s, shape={q_emb.shape}", flush=True)

    # search via numpy dot product (cosine similarity since both are normalized)
    print(f"  computing top-{TOP_K} via numpy...", flush=True)
    t1 = time.time()
    sims = q_emb @ chunk_emb.T  # (n_queries, n_chunks)
    indices = np.argpartition(-sims, TOP_K, axis=1)[:, :TOP_K]
    # sort top-k by score
    for i in range(indices.shape[0]):
        order = np.argsort(-sims[i, indices[i]])
        indices[i] = indices[i][order]
    print(f"  search done in {time.time()-t1:.2f}s", flush=True)

    # metrics
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--loaders", nargs="+", default=["all"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    loaders = list(LOADERS.keys()) if args.loaders == ["all"] else args.loaders
    gt = load_ground_truth()
    print(f"GT: {len(gt)} Q&A, target files: {len({g['target_file_name'] for g in gt})}")

    results = {}
    for name in loaders:
        if name not in LOADERS:
            print(f"unknown loader: {name}")
            continue
        out_path = OUT_DIR / f"{name}.json"
        if args.skip_existing and out_path.exists():
            print(f"\nSKIP {name} (cached)")
            results[name] = json.load(open(out_path))
            continue

        print(f"\n=== {name} ===")
        loader_out = process_loader(name, gt)
        if not loader_out["chunks"]:
            continue
        metrics = evaluate(loader_out["chunks"], loader_out["meta"], gt)
        result = {
            "loader": name,
            "n_pdfs": loader_out["n_ok"],
            "n_chunks": len(loader_out["chunks"]),
            "parse_time_sec": loader_out["elapsed"],
            "embed_time_sec": metrics["embed_time_sec"],
            "MRR": metrics["MRR"],
            "Hit@1": metrics["Hit@1"],
            "Hit@5": metrics["Hit@5"],
            "File@5": metrics["File@5"],
        }
        results[name] = result
        json.dump(result, open(out_path, "w"), ensure_ascii=False, indent=2)
        print(f"  MRR={metrics['MRR']:.4f} Hit@1={metrics['Hit@1']:.3f} Hit@5={metrics['Hit@5']:.3f} File@5={metrics['File@5']:.3f}")

    # summary
    print("\n" + "=" * 80)
    print(f"{'Loader':<18} {'PDFs':>5} {'Chunks':>8} {'Parse(s)':>9} {'MRR':>7} {'Hit@1':>7} {'Hit@5':>7} {'File@5':>7}")
    print("-" * 80)
    for name in loaders:
        r = results.get(name)
        if not r: continue
        print(f"{name:<18} {r['n_pdfs']:>5} {r['n_chunks']:>8} {r['parse_time_sec']:>9.1f} {r['MRR']:>7.4f} {r['Hit@1']:>7.3f} {r['Hit@5']:>7.3f} {r['File@5']:>7.3f}")


if __name__ == "__main__":
    main()
