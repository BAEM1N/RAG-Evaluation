# Phase 4 — Embedding Benchmark Leaderboard

**Date**: 2026-04-19
**Dataset**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
**Fixed**: parser=pymupdf4llm, chunk_size=500/100
**Config**: no truncation, llama.cpp `--pooling last` for harrier, default pooling for others, `-c 8192`

| Rank | Model | dim | MRR | p@1 | p@5 | f@5 | Notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 🥇 | **koe5** | 1024 | **0.6871** | 60.7% | 80.7% | 91.3% | Korean-specific |
| 🥈 | gemma-embed-300m | 768 | 0.6650 | 57.3% | 79.7% | 91.7% | Best tiny model |
| 🥉 | pixie-rune-v1 | 1024 | 0.6627 | 58.7% | 76.0% | 92.0% | |
| 4 | snowflake-arctic-ko | 1024 | 0.6612 | 58.3% | 75.0% | 91.7% | Korean-tuned |
| 5 | snowflake-arctic-l-v2 | 1024 | 0.6495 | 58.3% | 73.0% | 89.0% | |
| 6 | jina-v4-retrieval | 4096 | 0.6449 | 54.7% | 78.7% | 91.7% | |
| 7 | nomic-embed-v2-moe | 768 | 0.6435 | 56.7% | 75.3% | 90.0% | MoE |
| 8 | kure-v1 | 1024 | 0.6267 | 54.7% | 74.3% | 91.0% | Korean |
| 9 | harrier-0.6b | 1024 | 0.6131 | 53.3% | 70.3% | 88.7% | pooling=last |
| 10 | granite-278m | 768 | 0.5969 | 50.3% | 72.0% | 87.3% | IBM |
| 11 | me5-large-instruct | 1024 | 0.5882 | 50.7% | 70.7% | 90.7% | Multilingual-E5 |
| 12 | qwen3-embed-4b | 4096 | 0.5850 | 48.0% | 73.0% | 89.7% | |
| 13 | bge-m3 | 1024 | 0.5630 | 48.7% | 66.7% | 89.7% | |
| 14 | qwen3-embed-0.6b | 1024 | 0.5564 | 46.3% | 67.0% | 87.7% | |
| 15 | jina-v4-code | 4096 | 0.5334 | 42.3% | 67.7% | 88.0% | Code-tuned |
| 16 | harrier-270m | 640 | 0.5291 | 43.7% | 65.3% | 88.3% | pooling=last (mean was 0.5479) |
| 17 | qwen3-embed-8b | 4096 | 0.5271 | 44.3% | 64.7% | 86.3% | |
| 18 | granite-107m | 768 | 0.4786 | 38.0% | 60.3% | 83.0% | |
| 19 | nemotron-embed-8b | 4096 | 0.4617 | 36.3% | 59.0% | 88.0% | |
| 20 | llama-embed-nemotron-8b | 4096 | 0.4617 | 36.3% | 59.0% | 88.0% | Same as #19 |
| 21 | jina-v5-small-retrieval | 1024 | 0.3898 | 31.7% | 48.3% | 74.3% | |
| 22 | jina-code-1.5b | 1024 | 0.3248 | 23.0% | 46.3% | 82.0% | Code-tuned |
| 23 | e5-mistral-7b | 4096 | 0.2843 | 22.7% | 36.0% | 69.3% | English-biased |
| 24 | jina-v5-nano-matching | 512 | 0.1791 | 12.7% | 26.3% | 62.0% | Matching-tuned |
| 25 | mxbai-embed-large | 1024 | 0.1157 | 8.7% | 15.7% | 38.7% | English-only |
| 26 | labse | 768 | 0.0472 | 2.7% | 8.0% | 27.3% | Obsolete |
| 27 | harrier-27b | 5376 | 0.0170 | 1.0% | 2.3% | 15.7% | Korean unsuitable |

## Key Observations

1. **Small beats big for Korean**: koe5 (1024d) > qwen3-embed-8b (4096d) by 0.16 MRR
2. **Korean-specific wins**: koe5, snowflake-arctic-ko, kure-v1, pixie-rune-v1 all in top 8
3. **Harrier-27b collapse**: 97% of dims dead (variance < 0.0001), Korean query-doc separation broken
4. **Nemotron duplicate**: nemotron-embed-8b and llama-embed-nemotron-8b are identical (same architecture/weights)
5. **English-tuned models fail**: mxbai, labse, e5-mistral at bottom

## Experimental Fixes Applied

- **Truncation removed** (500-char cap unfairly penalized large models)
- **harrier-0.6b**: `--pooling last` per Microsoft spec → +0.094 MRR vs mean pooling
- **harrier-270m**: `--pooling last` tested, but mean was marginally better (0.5479 → 0.5291)
- **harrier-27b**: Both poolings measured, last slightly better (0.0033 → 0.0170)
- **qwen3-embed-8b**: Context size 8192 explicit — no change (0.5271 confirmed real)
