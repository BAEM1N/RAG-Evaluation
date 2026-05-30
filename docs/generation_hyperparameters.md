# Generation Hyperparameters

Phase 5A (generation) 호출 시 각 모델에 전달된 파라미터와, 명시하지 않은 필드가 어떤 provider default로 대체되었는지 기록.

## Common constants

| Variable | Value |
|---|---|
| Retrieval top-k | 5 |
| System/user prompt | `docs/experiment-design.md`의 `RAG_PROMPT` (ko) |
| Max output tokens | **제한 없음** (모든 provider에서 cap 생략) |
| Random seed | not set (provider default: system entropy) |

## OpenAI — `/v1/batches` → `/v1/chat/completions`

Explicit: `temperature: 0` (재현성)

| Model | we_sent | provider default used |
|---|---|---|
| gpt-5.4-pro | temperature=0 | top_p=1.0, freq_pen=0, pres_pen=0, n=1, reasoning_effort=high |
| gpt-5.4 | temperature=0 | top_p=1.0, freq_pen=0, pres_pen=0, reasoning_effort=medium |
| gpt-5.4-mini | temperature=0 | top_p=1.0, freq_pen=0, pres_pen=0 |
| gpt-5.4-nano | temperature=0 | top_p=1.0, freq_pen=0, pres_pen=0 |

## Anthropic — `/v1/messages/batches`

Explicit: `max_tokens=8192` (Anthropic 필수), `cache_control` on static prefix (ephemeral, 1h TTL)

Sampling 파라미터는 전부 **생략** — Opus 4.7 + extended thinking 조합에서 `temperature<1.0` 또는 `top_p<1.0`, `top_k`를 보내면 400 에러. 일관성을 위해 모든 Claude 모델에 동일 규칙 적용.

| Model | we_sent | provider default used |
|---|---|---|
| claude-opus-4-7 | max_tokens=8192 | temperature=1.0, top_p=-1, top_k=-1 |
| claude-sonnet-4-6 | max_tokens=8192 | temperature=1.0, top_p=-1, top_k=-1 |
| claude-sonnet-4-5 | max_tokens=8192 | temperature=1.0, top_p=-1, top_k=-1 |
| claude-haiku-4-5 | max_tokens=8192 | temperature=1.0, top_p=-1, top_k=-1 |
| claude-opus-4-7-thinking | max_tokens=8192, thinking.type=enabled | temperature=1.0 |
| claude-sonnet-4-6-thinking | max_tokens=8192, thinking.type=enabled | temperature=1.0 |

## Google — `batchGenerateContent`

Explicit: 생성은 config 전부 생략. Thinking variant만 `thinking_config.include_thoughts=True`.

| Model | we_sent | provider default used |
|---|---|---|
| gemini-3.1-pro-preview | — | temperature=1.0, top_p=0.95, top_k=64 |
| gemini-3.1-pro-preview-thinking | include_thoughts=True | temperature=1.0, top_p=0.95, top_k=64 |
| gemini-3-pro-preview | — | temperature=1.0, top_p=0.95, top_k=64 |
| gemini-3-flash-preview | — | temperature=1.0, top_p=0.95, top_k=64 |
| gemini-3.1-flash-lite-preview | — | temperature=1.0, top_p=0.95, top_k=64 |
| gemini-2.5-flash | — | temperature=1.0, top_p=0.95, top_k=64 |

## OpenRouter — realtime `/v1/chat/completions` (no batch API)

Sampling 전부 생략 (upstream default pass-through). Anthropic/Gemini-compatible 모델은 `cache_control: ephemeral` on prefix when >4KB.

| Model | Upstream | provider default used |
|---|---|---|
| x-ai/grok-4.20 | xAI | temperature=1.0, top_p=0.95 |
| moonshotai/kimi-k2.5 | Moonshot | temperature=0.6, top_p=1.0 |
| moonshotai/kimi-k2.6 | Moonshot | temperature=0.6, top_p=1.0 |
| minimax/minimax-m2.7 | MiniMax | temperature=1.0, top_p=0.95 |
| minimax/minimax-m2.5 | MiniMax | temperature=1.0, top_p=0.95 |
| qwen/qwen3-max-thinking | Alibaba | temperature=0.7, top_p=0.8, thinking=auto |
| qwen/qwen3.6-plus | Alibaba | temperature=0.7, top_p=0.8 |
| z-ai/glm-5.1 | Zhipu | temperature=0.8, top_p=0.95 |
| z-ai/glm-5 | Zhipu | temperature=0.8, top_p=0.95 |
| z-ai/glm-4.7 | Zhipu | temperature=0.8, top_p=0.95 |
| z-ai/glm-4.7-flash | Zhipu | temperature=0.8, top_p=0.95 |
| deepseek/deepseek-v4-pro | DeepSeek | temperature=0.7, top_p=1.0 |
| deepseek/deepseek-v4-flash | DeepSeek | temperature=0.7, top_p=1.0 |
| deepseek/deepseek-v3.2 | DeepSeek | temperature=0.7, top_p=1.0 |
| xiaomi/mimo-v2.5-pro | Xiaomi | temperature=0.7, top_p=0.9 |
| xiaomi/mimo-v2.5 | Xiaomi | temperature=0.7, top_p=0.9 |
| mistralai/mistral-small-2603 | Mistral | temperature=0.7, top_p=1.0 |
| upstage/solar-pro-3 | Upstage | temperature=0.8, top_p=0.95, top_k=50 |
| nvidia/nemotron-3-nano-30b-a3b | NVIDIA | temperature=0.6, top_p=0.95 |

## Local (reference)

로컬 generation (Phase 5A original, 12 LLMs) — ollama / llama.cpp:

| Knob | Value |
|---|---|
| temperature | 0 |
| top_k (sampling) | default (40 for llama.cpp, variable for ollama) |
| top_p | default |
| repeat_penalty | 1.1 (llama.cpp default) |
| num_predict | -1 (unlimited, `max_tokens` cap 없음) |
| thinking (Qwen3.5 변형) | `chat_template_kwargs.enable_thinking=False` (nothink) 명시, think variants 별도 |

## Cross-provider consistency note

- **Judge 호출**은 재현성을 위해 모든 provider에서 `temperature=0` 강제 (단 Anthropic은 sampling 파라미터 전부 생략, provider-side greedy equivalent는 없음).
- **Generation 호출**은 provider별 default 혼재. OpenAI만 `temperature=0` 명시 (API 특성 허용). 비교 시 이 차이를 명시.
- `cache_control` / `CachedContent` 적용은 판정(judge) 측에서 Q+metric prefix에만 적용. 생성 쪽은 provider auto-cache만 (OpenAI auto 1024+ tok, Google/OpenRouter provider default).

## Machine-readable version

`docs/generation_hyperparameters.json`에 동일 정보를 구조화해서 저장 (35 records, `we_sent` / `provider_defaults_used` / `endpoint` / `notes`).
