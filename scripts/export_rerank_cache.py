#!/usr/bin/env python3
"""Stage 4-2 cache export — chunks + queries + gt → JSON, top20 → npy.

GPU 서버에서 PDF 재파싱 없이 reranker bench 실행할 때 사용.
"""
import os, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_utils import parse_pdf_pymupdf, load_ground_truth

PDF_DIR = ROOT / "data/pdfs"
OUT_JSON = ROOT / "results/phase4_2_reranker/_chunks_cache.json"


def find_pdf(filename):
    for root, _dirs, files in os.walk(PDF_DIR):
        if filename in files:
            return Path(root) / filename
    return None


def main():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    gt = load_ground_truth()

    chunks_text, meta = [], []
    for fname in sorted({g["target_file_name"] for g in gt}):
        p = find_pdf(fname)
        if not p:
            print(f"  MISSING {fname}")
            continue
        for page in parse_pdf_pymupdf(str(p)):
            for part in splitter.split_text(page["text"]):
                part = part.strip()
                if len(part) >= 30:
                    chunks_text.append(part)
                    meta.append({"file": fname, "page": page["page"]})
    print(f"chunks: {len(chunks_text)}")

    out = {
        "chunks_text": chunks_text,
        "meta": meta,
        "queries": [g["question"] for g in gt],
        "gt": [{"target_file_name": g["target_file_name"], "target_page_no": str(g["target_page_no"])} for g in gt],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), ensure_ascii=False)
    print(f"saved JSON: {OUT_JSON} ({OUT_JSON.stat().st_size/1e6:.1f} MB)")
    print(f"top20 already at: {ROOT}/results/phase4_2_reranker/_retrieved_top20.npy")


if __name__ == "__main__":
    main()
