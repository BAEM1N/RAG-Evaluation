# Model Inventory

Complete list of models used in this benchmark, grouped by role (embedding, generation LLM, judge LLM) and by runtime (llama.cpp vs Ollama).

Local hosts referenced in this doc:

| Host | GPU / memory | Runtime |
|------|--------------|---------|
| GPU host (llama.cpp) | AMD MI100, 96 GB VRAM | llama-server / llama.cpp gateway |
| Unified-memory host (Ollama) | GB10, 128 GB unified memory | Ollama |
| Local Mac (Ollama) | — | Ollama |

All weights are GGUF unless noted otherwise.

---

## 1. Embedding Models (Phase 4, 27 configs — all local)

Format: GGUF (Q8_0 unless noted). Served via llama.cpp on the GPU host.

### Tier 1 — Large (7B+)

| alias | dim | Size | Phase 4 MRR | Notes |
|-------|----:|-----:|------------:|-------|
| qwen3-embed-8b | 4096 | 7.5 GB | 0.5325 | Qwen3, strong on MTEB multilingual |
| nemotron-embed-8b | 4096 | 7.5 GB | 0.4640 | NVIDIA |
| llama-embed-nemotron-8b | 4096 | 7.5 GB | — | same arch/weights as nemotron-embed-8b |
| e5-mistral-7b-instruct | 4096 | 7.2 GB | — | E5 on Mistral backbone |

### Tier 2 — Medium (1B–4B)

| alias | dim | Size | Phase 4 MRR | Notes |
|-------|----:|-----:|------------:|-------|
| qwen3-embed-4b | 4096 | 4.0 GB | 0.5862 | |
| jina-v4-retrieval | 4096 | 3.1 GB | 0.6489 | multimodal-capable |
| jina-v4-code | 4096 | 3.1 GB | 0.5442 | code-specialized |
| jina-code-1.5b | 1024 | 1.6 GB | 0.3288 | code-only |

### Tier 3 — Small, Korean-tuned (~300–600M)

| alias | dim | Size | Phase 4 MRR | Notes |
|-------|----:|-----:|------------:|-------|
| koe5 | 1024 | 417 MB | **0.6871** | Phase 4 rank 1 |
| snowflake-arctic-ko | 1024 | 605 MB | 0.6612 | |
| pixie-rune-v1 | 1024 | 605 MB | 0.6627 | |
| kure-v1 | 1024 | 606 MB | 0.6267 | |

### Tier 4 — Small, multilingual / general (~300–600M)

| alias | dim | Size | Phase 4 MRR | Notes |
|-------|----:|-----:|------------:|-------|
| snowflake-arctic-l-v2 | 1024 | 606 MB | 0.6495 | |
| nomic-embed-v2-moe | 768 | 489 MB | 0.6435 | MoE |
| qwen3-embed-0.6b | 1024 | 610 MB | 0.5621 | |
| harrier-0.6b | 1024 | 610 MB | 0.6131 | pooling=last |
| me5-large-instruct | 1024 | 576 MB | 0.5853 | multilingual |
| bge-m3 | 1024 | 606 MB | 0.5745 | dense+sparse+colbert hybrid |
| jina-v5-small-retrieval | 1024 | 610 MB | 0.3868 | |
| labse | 768 | 492 MB | 0.0472 | legacy, 109 languages |

### Tier 5 — Tiny (~100–300M)

| alias | dim | Size | Phase 4 MRR | Notes |
|-------|----:|-----:|------------:|-------|
| gemma-embed-300m | 768 | 314 MB | **0.6650** | Phase 4 rank 2; used as fixed embedding for Phase 5 judge runs |
| granite-278m | 768 | 290 MB | 0.5969 | IBM, Korean-capable |
| harrier-270m | 1024 | 279 MB | 0.5594 | |
| voyage-4-nano | 1024 | 355 MB | — | MoE, Apache 2.0 |
| mxbai-embed-large-v1 | 1024 | 342 MB | 0.1157 | English-only on this dataset |
| granite-107m | 768 | 116 MB | 0.4806 | |
| jina-v5-nano-matching | 512 | 223 MB | 0.1821 | |

### Outlier — Very Large

| alias | dim | Size | Phase 4 MRR | Notes |
|-------|----:|-----:|------------:|-------|
| harrier-27b | 5376 | 27 GB | 0.0170 | variance < 1e-4, ~97% dead dims on Korean → excluded from practical use |

### Self-converted GGUF

| Model | HuggingFace repo |
|-------|------------------|
| snowflake-arctic-embed-l-v2.0-ko | [BAEM1N/snowflake-arctic-embed-l-v2.0-ko-GGUF](https://huggingface.co/BAEM1N/snowflake-arctic-embed-l-v2.0-ko-GGUF) |
| PIXIE-Rune-v1.0 | [BAEM1N/PIXIE-Rune-v1.0-GGUF](https://huggingface.co/BAEM1N/PIXIE-Rune-v1.0-GGUF) |

Full Phase 4 leaderboard: `results/phase4_embedding/LEADERBOARD.md`.

---

## 2. Generation LLMs (Phase 5A, 12 configs)

### GPU host (llama.cpp)

| Model | Arch | Total params | Active | Quant | VRAM |
|-------|------|-------------:|-------:|-------|-----:|
| qwen3.5-27b | Dense | 27B | 27B | Q8_0 | 29 GB |
| qwen3.5-9b | Dense | 9B | 9B | Q4_K_M / Q8_0 | 6.6–10 GB |
| gpt-oss-120b | MoE | 120B | ~12B | MXFP4 | 65 GB |
| gpt-oss-20b | MoE | 20B | ~2B | MXFP4 | 13 GB |

### Unified-memory host (Ollama)

| Model | Arch | Total params | Active | Quant | Size |
|-------|------|-------------:|-------:|-------|-----:|
| qwen3.5:122b-a10b (nothink) | MoE | 122B | 10B | Q4_K_M | 81 GB |
| qwen3.5:122b-a10b (think) | MoE | 122B | 10B | Q4_K_M | 81 GB |
| deepseek-r1:70b | Dense | 70B | 70B | Q4 | 42 GB |
| exaone3.5:32b | Dense | 32B | 32B | default | 19 GB |
| mistral-small:24b | Dense | 24B | 24B | Q4 | 14 GB |
| lfm2:24b | Dense | 24B | 24B | Q4 | 14 GB |
| phi4:14b | Dense | 14B | 14B | Q4 | 9.1 GB |

---

## 3. Judge LLMs (Phase 5B, 6-judge ensemble)

| Judge | Host | Coverage |
|-------|------|----------|
| gemma4:31b (nothink) | Local Mac Ollama | full (12/12) |
| nemotron-120b (nothink) | GPU host, llama-server | full (12/12) |
| qwen3.5:122b-a10b (nothink) | Local Mac Ollama | full (12/12) |
| qwen3.6-35b-a3b (nothink) | GPU host, llama-server | full (12/12) |
| supergemma4-26b (nothink) | GPU host, llama-server | full (12/12) |
| qwen3-next:80b (nothink) | Unified-memory host, Ollama | partial |
| qwen3.5-27b-claude-distill (nothink) | GPU host, llama-server | partial |

Excluded: **solar-open-100b** — the ollama custom Modelfile's `TEMPLATE {{ .Prompt }}` skips the chat template, and the glm4moe backbone then runs in completion mode and never emits the 1–5 integer scores the rubric requires. Rerun path: `llama-server --jinja` using the GGUF's built-in chat template.

---

## 4. Optional: Hosted-API extension (27 configs)

The benchmark can be extended to hosted APIs for generation and / or judging; this is not required for Phases 1–5 but documents the budget envelope.

| Provider | Direct Batch API | Generation configs used |
|----------|:-:|------------------------|
| OpenAI | yes | gpt-5.4-pro, gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gpt-5.3-chat, o4-mini |
| Anthropic | yes | claude-opus-4.7, claude-sonnet-4.6, claude-sonnet-4.5, claude-haiku-4.5 |
| Google | yes | gemini-3.1-pro, gemini-3.1-flash, gemini-3.1-flash-lite, gemini-3-pro |
| OpenRouter | no | grok-4.20, kimi-k2.5, minimax-m2.7, qwen3-max-thinking, qwen3.6-plus, glm-5.1, sonar-reasoning-pro, command-a, mistral-large-3 |

See `docs/cost-report.md` for per-model pricing, judge caching math, and timeline.

---

## 5. Summary

| Role | Count | Location |
|------|------:|----------|
| Embedding | 27 | GPU host (llama.cpp) |
| Generation LLM | 12 | GPU host (4) + Unified-memory host (7) + mixed 9B (1) |
| Judge LLM | 7 (6 active + 1 excluded) | GPU host (3) + Local Mac (2) + Unified-memory host (1) + 1 excluded |
