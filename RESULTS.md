# RAG-Evaluation — Results Snapshot

**Dataset**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs, 5 domains)

Each Phase fixes the previous Phases' winners and varies a single component — single-variable comparison.

---

## Phase 1 — Parser (3)

| Parser | MRR | Hit@1 | Hit@5 | File@5 | Chunks | Notes |
|--------|----:|------:|------:|-------:|-------:|-------|
| **pymupdf4llm** | **0.4715** | 38.3% | 58.3% | 86.0% | 1,920 | markdown conversion (headers/tables) |
| pymupdf | 0.4663 | 35.7% | 63.3% | 86.3% | 1,263 | plain text |
| pypdf | 0.4472 | 34.3% | 60.7% | 82.7% | 1,224 | line-based |

Winner: **pymupdf4llm** (+5.4% MRR over pypdf).

---

## Phase 2 — Chunking (4)

Fixed: Parser=pymupdf4llm.

| Strategy | chunk_size / overlap | MRR | Hit@1 | Hit@5 | Chunks |
|----------|---------------------:|----:|------:|------:|-------:|
| **small** | 500 / 100 | **0.5315** | 45.0% | 65.0% | 3,166 |
| baseline | 1000 / 200 | 0.4713 | 38.3% | 58.3% | 1,920 |
| medium | 1500 / 200 | 0.4458 | 36.3% | 55.0% | 1,468 |
| large | 2000 / 300 | 0.4302 | 34.3% | 53.3% | 1,370 |

Winner: **500 / 100** (+23.5% MRR — the largest single-component impact observed).

---

## Phase 3 — VectorStore (7)

Fixed: Parser=pymupdf4llm, Chunking=500/100, Embedding=qwen3-embed-8b.

| Store | MRR | Hit@5 | Insert(s) | p95 latency (ms) | QPS |
|-------|----:|------:|----------:|-----------------:|----:|
| **FAISS** | 0.5304 | 65.0% | **0.76** | **0.74** | **1,394** |
| LanceDB | 0.5304 | 65.0% | 6.04 | 6.61 | 158 |
| Qdrant | 0.5304 | 65.0% | 58.58 | 122.20 | 9 |
| Milvus | 0.5304 | 65.0% | 22.39 | 57.47 | 19 |
| Weaviate | 0.5298 | 64.7% | 12.00 | 26.86 | 43 |
| Chroma | 0.5271 | 64.7% | 16.72 | 46.34 | 25 |
| pgvector | 0.5304 | 65.0% | 92.32 | 174.23 | 7 |

Winner: **FAISS** (accuracy tied top, ~200× faster p95).

---

## Phase 4 — Embedding (27)

Fixed: Parser=pymupdf4llm, Chunking=500/100, VectorStore=FAISS.

Top 10 + selected failures:

| Rank | Model | dim | MRR | Hit@1 | Hit@5 | Notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | **koe5** | 1024 | 0.6871 | 60.7% | 80.7% | Korean-tuned E5, 600M |
| 2 | gemma-embed-300m | 768 | 0.6650 | 57.3% | 79.7% | best tiny |
| 3 | pixie-rune-v1 | 1024 | 0.6627 | 58.7% | 76.0% | |
| 4 | snowflake-arctic-ko | 1024 | 0.6612 | 58.3% | 75.0% | Korean-tuned |
| 5 | snowflake-arctic-l-v2 | 1024 | 0.6495 | 58.3% | 73.0% | |
| 6 | jina-v4-retrieval | 4096 | 0.6449 | 54.7% | 78.7% | |
| 7 | nomic-embed-v2-moe | 768 | 0.6435 | 56.7% | 75.3% | MoE |
| 8 | kure-v1 | 1024 | 0.6267 | 54.7% | 74.3% | Korean |
| 9 | harrier-0.6b | 1024 | 0.6131 | 53.3% | 70.3% | pooling=last |
| 10 | granite-278m | 768 | 0.5969 | 50.3% | 72.0% | |
| … | (ranks 17–25) | | 0.5–0.1 | | | |
| 25 | mxbai-embed-large | 1024 | 0.1157 | 8.7% | 15.7% | English-only |
| 26 | labse | 768 | 0.0472 | 2.7% | 8.0% | legacy |
| 27 | harrier-27b | 5376 | 0.0170 | 1.0% | 2.3% | unsuitable for Korean |

Winner: **koe5** for generation / retrieval. **gemma-embed-300m** used as the fixed embedding for Phase 5 judge runs (rank 2, smaller, faster).

**Findings**:
- Korean-tuned small models outperform large English-centric ones (koe5 600M beats qwen3-embed-8b by 0.16 MRR).
- 4 Korean-tuned models (koe5, snowflake-ko, kure, pixie) are in the top 8.
- harrier-27b shows variance < 0.0001 with ~97% dead dimensions on this task — Korean query/document separation fails.
- nemotron-embed-8b and llama-embed-nemotron-8b are equivalent (same architecture/weights).

Full table: [results/phase4_embedding/LEADERBOARD.md](results/phase4_embedding/LEADERBOARD.md)

---

## Phase 5A — Generation LLMs (12)

Fixed: Parser/Chunking/VectorStore winners + Embedding=gemma-embed-300m, top-k=5.

| LLM | Host | Quant | Thinking |
|---|---|---|---|
| deepseek-r1:70b | Spark (ollama) | Q4 | nothink |
| exaone3.5:32b | Spark (ollama) | Q4 | — |
| gpt-oss:120b | AI-395 (llama.cpp) | MXFP4 | — |
| gpt-oss:20b | AI-395 (llama.cpp) | MXFP4 | — |
| lfm2:24b | Spark (ollama) | Q4 | — |
| mistral-small:24b | Spark (ollama) | Q4 | — |
| phi4:14b | Spark (ollama) | Q4 | — |
| qwen3.5:9b | Mixed | Q4_K_M / Q8_0 | nothink (2 configs) |
| qwen3.5:27b | AI-395 (llama.cpp) | Q8_0 | nothink |
| qwen3.5:122b-a10b | Spark (ollama) | Q4_K_M | nothink |
| qwen3.5:122b-a10b | Spark (ollama) | Q4_K_M | think |

---

## Phase 5B — LLM-as-Judge

**Method** (reproduction of the allganize rubric):
1. 4 metrics: similarity, correctness, completeness, faithfulness (MLflow v1 rubric, 1–5 scale).
2. threshold=4 → per-metric O/X.
3. Majority vote (≥ 2 metrics O → question counts as O).
4. Accuracy = O count / total questions.

### Judge matrix

| Judge | Host | Coverage |
|---|---|---|
| gemma4:31b (nothink) | Mac ollama | 12/12 |
| nemotron-120b (nothink) | AI-395 llama-server | 12/12 |
| qwen3.5:122b-a10b (nothink) | Mac ollama | 12/12 |
| qwen3.6-35b-a3b (nothink) | AI-395 llama-server | 12/12 |
| supergemma4-26b (nothink) | AI-395 llama-server | 12/12 |
| qwen3-next:80b (nothink) | DGX Spark ollama | partial (3/12) |
| qwen3.5-27b-claude-distill (nothink) | AI-395 llama-server | partial (0/12) |

### Excluded judge

**solar-open-100b** — excluded. Under the ollama custom Modelfile used (`TEMPLATE {{ .Prompt }}`), the chat template is bypassed and the glm4moe backbone runs in raw-completion mode, which does not produce the 1–5 integer scores the rubric requires. First pass returned 0.000 accuracy across all 12 candidate LLMs (1,200 votes, 100% parse failure). Rerunning via `llama-server --jinja` with the GGUF's built-in chat template (`<|begin|>role<|content|>msg<|end|>` with separate `<|think|>` blocks) is the correct path, but this judge is not included in the current snapshot.

### Current leaderboard (5-judge ensemble, partial)

Computed over the 5 judges with full 12/12 coverage. The two partial judges are not included in the aggregate below.

| Rank | LLM | Mean accuracy | Judge range |
|---:|---|---:|---|
| 1 | gpt-oss:120b | 0.77 | 0.73–0.80 |
| 2 | gpt-oss:20b | 0.76 | 0.71–0.81 |
| 3 | qwen3.5:122b-a10b (think) | 0.72 | 0.68–0.75 |
| 4 | qwen3.5:27b-q8 (nothink) | 0.72 | 0.67–0.74 |
| 5 | qwen3.5:122b-a10b (nothink) | 0.71 | 0.67–0.74 |
| 6 | exaone3.5:32b | 0.70 | 0.62–0.72 |
| 7 | phi4:14b | 0.70 | 0.62–0.73 |
| 8 | mistral-small:24b | 0.67 | 0.62–0.72 |
| 9 | qwen3.5:9b-q8 | 0.63 | 0.56–0.67 |
| 10 | qwen3.5:9b-q4 | 0.64 | 0.58–0.68 |
| 11 | deepseek-r1:70b | 0.62 | 0.36–0.72 (claude-distill outlier at 0.36) |
| 12 | lfm2:24b | 0.52 | 0.39–0.57 |

---

## Best Korean RAG configuration (current results)

```
Parser:       pymupdf4llm
Chunking:     chunk_size=500, overlap=100
VectorStore:  FAISS
Embedding:    koe5 (or gemma-embed-300m)
top-k:        5
LLM:          gpt-oss:120b (highest mean judge acc), or Qwen3.5:27b Q8 (top-tier with lower local VRAM)
```

---

## Reproduce

```bash
python scripts/bench_all.py --phase 1
python scripts/bench_all.py --phase 4 --model koe5
python scripts/phase5_batch_generate.py
python scripts/llm_judge.py <expB__*.json>
python scripts/judge_leaderboard.py
```

Experiment design: [docs/experiment-design.md](docs/experiment-design.md)
Hosted-API cost extension: [docs/cost-report.md](docs/cost-report.md)
