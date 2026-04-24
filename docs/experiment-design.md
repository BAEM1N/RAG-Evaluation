# Experiment Design

> Korean RAG pipeline component benchmark — research questions, variables, metrics, and protocol.

## 1. Research Questions

This project quantifies **how much each RAG pipeline component contributes to final quality** on a Korean dataset.

- RQ1. How does the PDF parser affect retrieval quality?
- RQ2. How do chunk size / overlap affect retrieval accuracy?
- RQ3. Do 7 vector stores differ meaningfully in accuracy or latency?
- RQ4. How do Korean-capable embedding models compare across sizes / architectures?
- RQ5. How does the generation LLM (local quantized vs. API) affect answer quality?
- RQ6. Does thinking / reasoning mode improve answer quality?

## 2. Dataset

[allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)

| Field | Value |
|-------|-------|
| Questions | 300 |
| Domains | finance / public / medical / law / commerce (60 each) |
| PDFs | 58 |
| Context type | paragraph 148 / image 57 / table 50 / text 45 |

Every question carries a gold PDF file name, page number, and answer text.

## 3. Phased Structure

Each Phase **fixes every component except one** and sweeps the variable. The winner of each Phase becomes the fixed value for the next.

```
[baseline]  PyPDFLoader → RecursiveSplit(1000/200) → OpenAIEmbedding → Chroma → gpt-4-turbo

Phase 1: {pypdf, pymupdf4llm, pymupdf}                       → rest fixed
Phase 2: (Phase 1 winner) → {500, 1000, 1500, 2000} × overlap
Phase 3: (Phase 1~2 winner) → {pgvector, FAISS, Chroma, Milvus, Qdrant, Weaviate, LanceDB}
Phase 4: (Phase 1~3 winner) → {27 embedding models}
Phase 5: (Phase 1~4 winner) → {12 generation LLMs}, judged by a 6-judge ensemble
```

### Experiment scale

| Phase | Variable | Configs | Queries |
|-------|----------|--------:|--------:|
| 1 Parser | 3 | 3 | 900 |
| 2 Chunking | 4 | 4 | 1,200 |
| 3 VectorStore | 7 | 7 | 2,100 |
| 4 Embedding | 27 | 27 | 8,100 |
| 5A Generation LLM | 12 | 12 | 3,600 |
| 5B Judge (4 metrics × 12 candidates × 6 judges) | 72 | 72 | 86,400 |
| **Total** | | **125** | **~102,300** |

## 4. Fixed Values (Baseline Frozen)

Every Phase starts from this baseline and changes exactly one component.

| Component | Value | Source |
|-----------|-------|--------|
| Parser | pymupdf4llm | Phase 1 MRR winner (0.4715) |
| Chunking | 500 / overlap 100 | Phase 2 MRR winner (0.5315) |
| VectorStore | FAISS | Phase 3 speed winner (accuracy tied) |
| Embedding (Phase 5) | gemma-embed-300m | Phase 4 rank 2 MRR (0.6650) — smaller, faster than rank 1 for batch judging |
| top-k | 5 | close to allganize original k=6 |
| Truncate | 500 chars | llama.cpp 512-token embedding limit |

## 5. Metrics

### Retrieval (Phase 1–4)

| Metric | Definition | Role |
|--------|------------|------|
| Hit@1 | top-1 matches the gold page | top-rank accuracy |
| Hit@5 | gold page in top-5 | practical retrieval accuracy |
| File Hit@5 | gold file in top-5 | file-level accuracy |
| **MRR** | mean reciprocal rank of gold chunk | primary metric (rank-aware) |
| NDCG@5 | rank-discounted gain | supplementary |

### Generation (Phase 5)

Phase 5B reproduces the allganize evaluation rubric with an ensemble of local LLM judges.

| Metric | Definition | Scale |
|--------|------------|-------|
| answer_similarity | semantic similarity vs. gold answer | 1–5 |
| answer_correctness | factual consistency vs. gold | 1–5 |
| completeness | coverage of key points in the gold answer | 1–5 |
| faithfulness | absence of hallucination vs. retrieved context | 1–5 |

**Aggregation**: per metric, threshold = 4 → O/X; ≥ 2 metrics O → the question counts as O for that candidate. Per-candidate accuracy = O count / 300.

**Judge ensemble**: 6 local LLM judges (see `RESULTS.md` § Phase 5B). Final leaderboard reports the mean and range across judges with full coverage.

## 6. Phase 5 Scope

Phase 5 is a **single-variable generation sweep**: 1 fixed embedding (gemma-embed-300m) × 12 generation LLM configs × 300 questions, judged by the 6-judge ensemble. Embedding × LLM interaction is not explored at this stage.

## 7. Host Assignment (generation + judge)

| Host | Runtime | Models |
|------|---------|--------|
| AI-395 (AMD MI100 96GB VRAM) | llama.cpp / llama-server | qwen3.5-27b Q8, qwen3.5-9b, EXAONE, Midm, SuperGemma4-26b, nemotron-120b (judge), qwen3.6-35b-a3b (judge), qwen3.5-27b-claude-distill (judge) |
| DGX Spark (GB10, 128GB unified) | Ollama | qwen3.5-122b-a10b (think/nothink), qwen3.5-27b, qwen3.5-9b, gpt-oss 20b/120b, deepseek-r1:70b, exaone3.5:32b, phi4:14b, mistral-small:24b, lfm2:24b, qwen3-next:80b (judge) |
| Mac (local) | Ollama | gemma4:31b (judge), qwen3.5:122b-a10b (judge) |

## 8. Parallelism

### Server side

- llama.cpp: `-np 8` (8 slots)
- Ollama: `OLLAMA_NUM_PARALLEL=8`

### Client side

- AI-395 scripts: `ThreadPoolExecutor(max_workers=4)`
- Spark scripts: `ThreadPoolExecutor(max_workers=8)`

## 9. Time / Cost

### Wall-clock (measured)

| Phase | Time |
|-------|------|
| 1 Parser | ~1 h |
| 2 Chunking | ~1 h |
| 3 VectorStore | ~10 min (embedding reused once) |
| 4 Embedding (27) | ~6 h (local) |
| 5A Generation (12 configs) | 1–2 days (AI-395 + Spark in parallel) |
| 5B Judge (6 judges × 12 candidates × 300 × 4 metrics) | 3–5 days |

### Cost

- Phases 1–5 as run: **$0** (fully local).
- Optional hosted-API extension: see `docs/cost-report.md`.

## 10. Reproduction

```bash
# Env
uv sync

# Phase 1–4 (local)
python scripts/bench_all.py --phase 1
python scripts/bench_all.py --phase 2 --parser pymupdf4llm
python scripts/bench_all.py --phase 3 --parser pymupdf4llm --chunk-size 500 --chunk-overlap 100
python scripts/bench_phase4_parallel.py

# Precompute retrieval cache
python scripts/precompute_retrieval.py

# Phase 5A (generation)
python scripts/phase5_batch_generate.py

# Phase 5B (judge)
python scripts/llm_judge.py <expB__*.json>
python scripts/judge_leaderboard.py
```

## 11. Known Issues and Mitigations

| Issue | Cause | Mitigation |
|-------|-------|------------|
| Empty embedding responses | llama.cpp 512-token embedding limit | 500-char truncate |
| Qwen3.5 thinking token waste | thinking defaults ON | `chat_template_kwargs: {enable_thinking: False}` |
| pgvector index build fails | 4,096-dim > HNSW 2,000-dim limit | sequential scan when dim > 2,000 |
| First-request latency (Ollama) | cold model load 30s+ | pre-warm with `ollama list` |
| harrier-27b incoherent output | GGUF quantization on this arch | excluded from leaderboard |
| solar-open-100b parse failure | ollama custom Modelfile empty `TEMPLATE` skipped the chat template | excluded; rerun path is `llama-server --jinja` with GGUF's built-in template |

## 12. Public artifacts

- `data/ground_truth.json` — 300 Q&A with file/page mapping
- `data/prepared_chunks.json` — chunks for the winning parser/chunking
- `data/retrieval_cache/` — 27 embeddings × 300 questions retrieval cache
- `results/phase1_*` to `results/phase5_*` — per-Phase JSON results
- `results/retrieval_analysis/` — cache-derived analysis CSVs
- `scripts/*.py` — all benchmark code
- `docs/*.md` — experiment design, cost report, model inventory
