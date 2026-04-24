# Cost Report — Extending the Benchmark to Hosted APIs

This document is a cost breakdown for reproducing the benchmark when generation and/or judging is extended to hosted APIs (OpenAI, Anthropic, Google, OpenRouter). All locally-run Phase 1–5 stages cost $0. The numbers below are list-price estimates — actual spend depends on provider billing at run time.

Token cost basis: Korean 2.5 chars/token.

---

## 1. Input Data Scale

| Item | Value |
|---|---|
| Questions | 300 |
| Chunks | 3,166 |
| Domains | 5 (finance, public, medical, law, commerce) |
| Fixed embedding (generation/judging) | gemma-embed-300m (Phase 4 rank 2) |
| Avg generation input | 923 tokens (min 376 / max 2,991) |
| Avg generation output | 100 tokens (3 Korean sentences) |
| Avg judge input | 385 tokens (static 255 + candidate 100) |
| Judge output | 1 token |

---

## 2. Generation LLMs — 27 hosted configs

### 2.1 OpenAI Direct API (6)

| Model | Batch $/1M (in/out) | 300 Q&A |
|---|---|---|
| gpt-5.4-pro (thinking) | $15 / $90 | **$6.85** |
| gpt-5.4 (thinking) | $1.25 / $7.50 | $0.58 |
| gpt-5.4-mini | $0.125 / $1.00 | $0.07 |
| gpt-5.4-nano | $0.025 / $0.20 | $0.01 |
| gpt-5.3-chat (nothink) | ~$1.00 / $5.00 | $0.42 |
| o4-mini (reasoning) | $0.55 / $2.20 | $0.22 |
| **Subtotal** | | **$8.15** |

### 2.2 Anthropic Direct API (4)

| Model | Batch $/1M | 300 Q&A |
|---|---|---|
| claude-opus-4.7 | $2.50 / $12.50 | **$1.07** |
| claude-sonnet-4.6 | $1.50 / $7.50 | $0.64 |
| claude-sonnet-4.5 | $1.50 / $7.50 | $0.64 |
| claude-haiku-4.5 | $0.50 / $2.50 | $0.22 |
| **Subtotal** | | **$2.57** |

### 2.3 Google Direct API (4)

| Model | Batch $/1M | 300 Q&A |
|---|---|---|
| gemini-3.1-pro | $1.00 / $6.00 | **$0.46** |
| gemini-3.1-flash | $0.25 / $1.50 | $0.11 |
| gemini-3.1-flash-lite | $0.125 / $0.75 | $0.06 |
| gemini-3-pro (legacy) | ~$0.625 / $5 | $0.33 |
| **Subtotal** | | **$0.96** |

### 2.4 Thinking variants (4, output ×3 assumed)

| Model | 300 Q&A |
|---|---|
| gpt-5.4 (think) | ~$1.00 |
| claude-opus-4.7 (think) | ~$3.20 |
| gemini-3.1-pro (think) | ~$1.40 |
| claude-sonnet-4.6 (think) | ~$1.90 |
| **Subtotal** | **$7.50** |

### 2.5 OpenRouter (9, non-direct providers)

Batch API unsupported → standard pricing. Partial caching on some routes.

| Model | Rate estimate | 300 Q&A |
|---|---|---|
| xai/grok-4.20 | ~$3/$15 | ~$1.30 |
| moonshotai/kimi-k2.5 (1T) | ~$2/$10 | ~$0.90 |
| minimax/minimax-m2.7 (230B) | ~$1/$5 | ~$0.45 |
| qwen/qwen3-max-thinking | ~$1.60/$6.40 | ~$0.80 |
| qwen/qwen3.6-plus | ~$0.80/$3.20 | ~$0.40 |
| z-ai/glm-5.1 | ~$1/$4 | ~$0.40 |
| perplexity/sonar-reasoning-pro | ~$2/$8 | ~$0.80 |
| cohere/command-a | ~$2.50/$10 | ~$1.00 |
| mistralai/mistral-large-3 | ~$2/$8 | ~$0.80 |
| **Subtotal** | | **~$6.85** |

### Generation total

| Group | Cost |
|---|---|
| Direct API (18 configs) | $19.18 |
| OpenRouter (9 configs) | $6.85 |
| **Generation total (27 configs)** | **~$26** |

---

## 3. Judging — Flagship 3-judge ensemble

**Judge targets**: 12 local + 27 hosted = **39 generation configs**
**Judge calls**: 4 metrics × 300 Q&A × 39 LLMs = **46,800 calls per judge**
**Judge input tokens**: 385 tokens × 46,800 = **18.0M tokens per judge**

### 3.1 Without prompt caching (batch only)

| Judge | Batch $/1M | Input | Output | Subtotal |
|---|---|---|---|---|
| gpt-5.4-pro | $15 / $90 | $270 | $4.2 | **$274** |
| claude-opus-4.7 | $2.50 / $12.50 | $45 | $0.6 | **$46** |
| gemini-3.1-pro | $1 / $6 | $18 | $0.3 | **$18** |
| **3-judge total** | | | | **~$338** |

### 3.2 With prompt caching

Static portion (template + question + target) = 255/385 = **66% cacheable**. Same question × 27 varying candidate answers → cache-friendly.

| Judge | Cached discount | After cache |
|---|---|---|
| gpt-5.4-pro (OpenAI auto cache) | 50% off cached | **~$150** |
| claude-opus-4.7 (explicit cache 0.1×) | 90% off cached | **~$15** |
| gemini-3.1-pro (context cache 0.2×) | 80% off cached | **~$8** |
| **3-judge total (cached)** | | **~$173** |

### Caching impact

- Batch only: $338
- Batch + caching: **$173** (−50%)

---

## 4. Embedding API extension (Phase 4)

Total text: 3,166 chunks + 300 questions = **3,466 texts**, avg 200 tokens ≈ **700K tokens**.

| Model | dim | $/1M | Cost |
|---|---|---|---|
| OpenAI text-embedding-3-small | 1536 | $0.02 | **$0.015** |
| OpenAI text-embedding-3-large | 3072 | $0.13 | **$0.09** |
| Google gemini-embedding-001 | 768/1536/3072 | $0.025 | **$0.018** |
| Google text-multilingual-embedding-002 | 768 | $0.025 | **$0.018** |
| **Embedding total** | | | **$0.14** |

Extension: **27 local + 4 API = 31 embeddings**.

---

## 5. Full-run cost summary

| Item | Cost |
|---|---|
| Generation 27 configs | $26 |
| Embedding API 4 | $0.14 |
| 3-judge flagship (batch + caching) | $173 |
| **Total** | **~$200** |

Recommended buffer: +20% for re-submissions and provider errors → **~$240** upper bound.

### Sensitivity

- Worst case (cache hit rate 0%, no batch): ~$340
- Best case (cache hit rate 90%): ~$150
- Excluding the most expensive judge (gpt-5.4-pro) and substituting gpt-5.4 standard: ~$100
- Direct API only (no OpenRouter): ~$193

---

## 6. Execution timeline

| Stage | Wall-clock | Note |
|---|---|---|
| Direct API batch submit | 24h | OpenAI / Claude async batches |
| OpenRouter realtime | ~3–5h | 9 LLMs × 300 Q&A sequential |
| Embedding API | ~30min | 4 models × 3,466 texts |
| Judge batch | 24–48h | 3 flagships × caching |
| **Total** | **~48–72h** | |

---

## 7. API key requirements

| Provider | Env var | Used for |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | 6 generation + 2 embedding + 1 judge |
| Anthropic | `ANTHROPIC_API_KEY` | 4 generation + 1 judge |
| Google | `GOOGLE_API_KEY` or Vertex SA JSON | 4 generation + 2 embedding + 1 judge |
| OpenRouter | `OPENROUTER_API_KEY` | 9 generation (non-direct providers) |

---

## 8. Risks

- **Thinking-mode output blowup**: some reasoning models emit 2,000+ tokens → cost ×5–10. Cap with `num_predict` / `max_tokens` 500–1000.
- **Batch failures**: provider-side errors require re-submission → keep 20% buffer.
- **OpenRouter markup**: some routes add 5–10% on top of upstream list price.
- **Cache TTL expiry**: Anthropic 5min / OpenAI 10min — order judge calls so the same question's candidates are batched contiguously to maximize hit rate.
