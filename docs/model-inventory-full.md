# RAG 벤치마크 전체 모델 인벤토리

> 최종 업데이트: 2026-04-16  
> 데이터셋: allganize/RAG-Evaluation-Dataset-KO (300 Q&A, 5 도메인)

---

## 1. 임베딩 모델 (Phase 4, 21개 — 전부 로컬)

서버: AI-395 (192.168.50.245), MI100 96GB VRAM  
게이트웨이: `http://192.168.50.245:8080/v1`  
형식: GGUF (Q8_0)

### Tier 1: 대형 (7B+)

| # | alias | dim | 크기 | MRR | 비고 |
|---|-------|-----|------|-----|------|
| 1 | qwen3-embed-8b | 4096 | 7.5GB | 0.5325 | Qwen3, MTEB 상위 |
| 2 | nemotron-embed-8b | 4096 | 7.5GB | 0.4640 | NVIDIA |
| 3 | jina-v4-retrieval | 4096 | 3.1GB | **0.6489** | 멀티모달 가능 |
| 4 | jina-v4-code | 4096 | 3.1GB | 0.5442 | 코드 특화 |
| 5 | qwen3-embed-4b | 4096 | 4.0GB | 0.5862 | |

### Tier 2: 소형 — 한국어 특화 (~335M)

| # | alias | dim | 크기 | MRR | 비고 |
|---|-------|-----|------|-----|------|
| 6 | kure-v1 | 1024 | 606MB | **0.6412** | 한국어 4위 |

### Tier 3: 소형 — 다국어/범용 (~335M)

| # | alias | dim | 크기 | MRR | 비고 |
|---|-------|-----|------|-----|------|
| 7 | snowflake-arctic-l-v2 | 1024 | 606MB | **0.6489** | |
| 8 | nomic-embed-v2-moe | 768 | 489MB | **0.6484** | MoE |
| 9 | bge-m3 | 1024 | 606MB | 0.5745 | 다국어, hybrid |
| 10 | me5-large-instruct | 1024 | 576MB | 0.5853 | 다국어 |
| 11 | qwen3-embed-0.6b | 1024 | 610MB | 0.5621 | |
| 12 | harrier-0.6b | 1024 | 610MB | 0.5266 | MS |
| 13 | labse | 768 | 492MB | 0.0468 | Google 109개 언어 |
| 14 | jina-v5-small-retrieval | 1024 | 610MB | 0.3868 | |
| 15 | jina-code-1.5b | 1024 | 1.6GB | 0.3288 | 코드 전용 |

### Tier 4: 초소형 (~100M)

| # | alias | dim | 크기 | MRR | 비고 |
|---|-------|-----|------|-----|------|
| 16 | **gemma-embed-300m** | 768 | 314MB | **0.6682** ⭐ | **MRR 1위**, Google |
| 17 | granite-278m | 768 | 290MB | 0.5973 | IBM, 한국어 |
| 18 | harrier-270m | 1024 | 279MB | 0.5594 | MS |
| 19 | granite-107m | 768 | 116MB | 0.4806 | IBM |
| 20 | jina-v5-nano-matching | 512 | 223MB | 0.1821 | |

### Tier 5: 초대형

| # | alias | dim | 크기 | MRR | 비고 |
|---|-------|-----|------|-----|------|
| 21 | harrier-27b | 4096 | 27GB | 0.0044 | MS, GGUF 양자화 문제 추정 |

### Phase 4 미완료 (게이트웨이 503 실패, 7개)

| # | alias | dim | 크기 | 상태 |
|---|-------|-----|------|------|
| - | snowflake-arctic-ko | 1024 | 605MB | 503 실패 |
| - | pixie-rune-v1 | 1024 | 605MB | 503 실패 |
| - | koe5 | 1024 | 417MB | 503 실패 |
| - | mxbai-embed-large | 1024 | 342MB | 503 실패 |
| - | voyage-4-nano | 1024 | 355MB | 503 실패 |
| - | llama-embed-nemotron-8b | 4096 | 7.5GB | 503 실패 |
| - | e5-mistral-7b | 4096 | 7.2GB | 503 실패 |

---

## 2. LLM 모델 — 로컬 보유

### AI-395 (llama.cpp 게이트웨이, MI100 96GB VRAM)

| 모델 | 아키텍처 | 총 파라미터 | Active | 양자화 | VRAM | 상시 |
|------|---------|-----------|--------|--------|------|------|
| qwen3.5-27b | Dense | 27B | 27B | Q4_K_M | 26GB | O |
| qwen3.5-35b-a3b | MoE | 35B | 3B | UD-IQ4_XS | 20GB | O |
| qwen3.5-9b | Dense | 9B | 9B | Q4_K_M | 8GB | 온디맨드 |
| nemotron-3-super-120b | MoE | 120B | 12B | UD-IQ4_XS | 80GB | 온디맨드 |

**다운로드 완료:**

| 모델 | 회사 | 크기 | 출처 |
|------|------|------|------|
| EXAONE 4.5-33B | LG AI | Q4_K_M 19GB | LGAI-EXAONE/EXAONE-4.5-33B-GGUF |
| Midm 2.0 Base 11.5B | KT | Q4_K_M 6.6GB | K-intelligence/Midm-2.0-Base-Instruct-GGUF |

**다운로드 중 (커뮤니티 파인튜닝):**

| 모델 | 원본 | Q4_K_M | 출처 | 특징 |
|------|------|--------|------|------|
| **SuperGemma 4-26B v2** | Gemma 4-26B MoE(4B active) | 16.8GB | Jiunsong/supergemma4-26b-uncensored-gguf-v2 | uncensored, Fast 라인 기반, 한국어 속도 89 tok/s |
| **Qwen3.5-27B Claude 4.6 Opus 증류** | Qwen3.5-27B | 16.5GB | Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF | Claude Opus reasoning CoT 지식증류 (SFT+LoRA) |

### DGX Spark (Ollama, GB10 128GB unified, 3.7TB SSD)

**설치 완료:**

| 모델 | 회사 | 총 파라미터 | Active | 양자화 | 크기 |
|------|------|-----------|--------|--------|------|
| qwen3.5:122b-a10b | Alibaba | 122B | 10B | Q4_K_M | 81GB |
| qwen3.5:35b-a3b | Alibaba | 35B | 3B | Q8_0/Q4_K_M | 38/23GB |
| qwen3.5:27b | Alibaba | 27B | 27B | Q8_0/Q4_K_M | 29/17GB |
| qwen3.5:9b | Alibaba | 9B | 9B | Q8_0/Q4_K_M | 10/6.6GB |
| exaone3.5:32b | LG AI | 32B | 32B | default | 19GB |
| exaone3.5:7.8b | LG AI | 7.8B | 7.8B | default | 4.8GB |
| ingu627/exaone4.0:32b | LG AI | 32B | 32B | default | 19GB |
| gpt-oss:120b | OpenAI | 120B MoE | ~12B | MXFP4 | 65GB |
| gpt-oss:20b | OpenAI | 20B MoE | ~2B | MXFP4 | 13GB |
| deepseek-r1:70b | DeepSeek | 70B | 70B | Q4 | 42GB |
| phi4:14b | Microsoft | 14B | 14B | Q4 | 9.1GB |
| mistral-small:24b | Mistral | 24B | 24B | Q4 | 14GB |
| lfm2:24b | Liquid AI | 24B | 24B | Q4 | 14GB |

**다운로드 중 (Spark):**

| 모델 | 회사 | 크기 | 특징 |
|------|------|------|------|
| gemma4:31b | Google | 20GB | 2026.04 신규, 멀티모달, Apache 2.0 |
| gemma4:26b (MoE 4B active) | Google | 18GB | 효율적 MoE |
| gemma4 (E4B) | Google | 9.6GB | 경량 |
| qwen3-next:80b | Alibaba | 50GB | Qwen3 차세대 MoE |
| llama4:scout (10M ctx) | Meta | 67GB | 2026.04, 초장문 컨텍스트 |
| solar-pro:latest | Upstage | 13GB | 한국어 MMLU 상위 |
| qwen3:latest | Alibaba | 5.2GB | Qwen3 (v3) |

---

## 3. LLM 모델 — 프로바이더 전용 (OpenRouter)

> 로컬 구동 불가 또는 오픈소스 미제공 모델  
> 실험 B (임베딩 1개 고정 × N LLM 비교)에서 사용  
> 기준: OpenRouter API (2026-04-16 가격)

### Tier S: 프리미엄 (최고 성능)

| 모델 | OpenRouter ID | Ctx | Input $/M | Output $/M | 300건 예상 |
|------|-------------|-----|-----------|-----------|-----------|
| GPT-5.4 Pro | openai/gpt-5.4-pro | 1.05M | $30.00 | $180.00 | ~$63.0 |
| Claude Opus 4.6 | anthropic/claude-opus-4.6 | 1M | $5.00 | $25.00 | ~$8.8 |
| Gemini 3.1 Pro | google/gemini-3.1-pro-preview | 1M | $2.00 | $12.00 | ~$4.2 |

### Tier A: 고성능

| 모델 | OpenRouter ID | Ctx | Input $/M | Output $/M | 300건 예상 |
|------|-------------|-----|-----------|-----------|-----------|
| GPT-5.4 | openai/gpt-5.4 | 1.05M | $2.50 | $15.00 | ~$5.3 |
| Claude Sonnet 4.6 | anthropic/claude-sonnet-4.6 | 1M | $3.00 | $15.00 | ~$5.3 |
| Qwen3.6-Plus | qwen/qwen3.6-plus | 1M | $0.33 | $1.95 | ~$0.7 |

### Tier B: 가성비

| 모델 | OpenRouter ID | Ctx | Input $/M | Output $/M | 300건 예상 |
|------|-------------|-----|-----------|-----------|-----------|
| GPT-5.4-mini | openai/gpt-5.4-mini | 400K | $0.75 | $4.50 | ~$1.6 |
| GPT-5.4-nano | openai/gpt-5.4-nano | 400K | $0.20 | $1.25 | ~$0.4 |
| Claude Sonnet 4.6 (fast) | anthropic/claude-sonnet-4.6 | 1M | $3.00 | $15.00 | ~$5.3 |
| Gemini 3.1 Flash Lite | google/gemini-3.1-flash-lite | 1M | $0.25 | $1.50 | ~$0.5 |
| Qwen3.5-Flash | qwen/qwen3.5-flash | 1M | $0.065 | $0.26 | ~$0.1 |

### Tier K: 한국어 특화 (프로바이더)

| 모델 | OpenRouter ID | Ctx | Input $/M | Output $/M | 300건 예상 |
|------|-------------|-----|-----------|-----------|-----------|
| Solar Pro 3 | upstage/solar-pro-3 | 128K | $0.15 | $0.60 | ~$0.2 |

### Tier D: DeepSeek / Mistral / Grok

| 모델 | OpenRouter ID | Ctx | Input $/M | Output $/M | 300건 예상 |
|------|-------------|-----|-----------|-----------|-----------|
| **Grok 4.20** | x-ai/grok-4.20 | 2M | $2.00 | $6.00 | ~$2.5 |
| DeepSeek V3.1 (Nex) | nex-agi/deepseek-v3.1-nex-n1 | 131K | $0.14 | $0.50 | ~$0.2 |
| Mistral Small | mistralai/mistral-small-2603 | 262K | $0.15 | $0.60 | ~$0.2 |
| Devstral (코딩) | mistralai/devstral-2512 | 262K | $0.40 | $2.00 | ~$0.7 |

### Tier C: 중국/아시아 프로바이더

| 모델 | OpenRouter ID | Ctx | Input $/M | Output $/M | 300건 예상 |
|------|-------------|-----|-----------|-----------|-----------|
| **GLM-5.1** | z-ai/glm-5.1 | 202K | $0.95 | $3.15 | ~$1.2 |
| GLM-4.7 | z-ai/glm-4.7 | 202K | $0.39 | $1.75 | ~$0.6 |
| **MiniMax M2.7** | minimax/minimax-m2.7 | 196K | $0.30 | $1.20 | ~$0.5 |
| MiniMax M2.5 | minimax/minimax-m2.5 | 196K | $0.12 | $0.99 | ~$0.3 |
| **Kimi K2.5** | moonshotai/kimi-k2.5 | 262K | $0.38 | $1.72 | ~$0.6 |

### Tier F: Friendli.ai 고유 모델 (OpenRouter에 없음)

> Friendli.ai는 한국 스타트업 프로바이더, LG AI와 독점 제휴 (K-EXAONE)  
> Endpoint: `https://api.friendli.ai/serverless/v1` (OpenAI 호환)  
> Prompt caching 지원 (대부분 모델 50~75% 할인)

| 모델 | Friendli ID | Input $/M | Cached $/M | Output $/M | 300건 예상 | 특징 |
|------|-------------|-----------|-----------|-----------|-----------|------|
| **K-EXAONE 236B-A23B** ⭐ | `LGAI-EXAONE/K-EXAONE-236B-A23B` | $0.20 | $0.10 | $0.80 | ~$0.3 | **Friendli 독점**, 한국어 세계 10위 |
| **Qwen3-235B-A22B-Instruct** ⭐ | `Qwen/Qwen3-235B-A22B-Instruct-2507` | $0.20 | - | $0.80 | ~$0.3 | Qwen3 v3 (Qwen3.5 아님) |
| **DeepSeek V3.2** ⭐ | `deepseek-ai/DeepSeek-V3.2` | $0.50 | $0.25 | $1.50 | ~$0.6 | OR은 V3.1까지만 |
| **Llama 3.3-70B-Instruct** ⭐ | `meta-llama/Llama-3.3-70B-Instruct` | $0.60 (unified) | - | $0.60 | ~$0.3 | 공식 Meta API |
| **Llama 3.1-8B-Instruct** | `meta-llama/Llama-3.1-8B-Instruct` | $0.10 (unified) | - | $0.10 | ~$0.05 | 기준 비교용 |

> OpenRouter와 중복되는 Friendli 모델 (제외): MiniMax M2.5, GLM-5/5.1, DeepSeek V3.1

### 전체 프로바이더 비용 시나리오

| 시나리오 | 모델 수 | 총 비용 |
|---------|--------|---------|
| 프리미엄 3개 (GPT-5.4 Pro, Opus 4.6, Gemini 3.1 Pro) | 3 | ~$76 |
| 고성능 5개 (GPT-5.4, Sonnet 4.6, Grok 4.20, Qwen3.6+, GLM-5.1) | 5 | ~$15 |
| 가성비 8개 (mini, nano, Flash, DeepSeek, Mistral, MiniMax, Kimi, Solar3) | 8 | ~$5 |
| Friendli 고유 5개 (K-EXAONE, Qwen3-235B, DS V3.2, Llama 3.3/3.1) | 5 | ~$1.5 |
| **전체 프로바이더 (OR + Friendli)** | **~27** | **~$103** |

> ※ 300건 비용 = (input ~1,000 tok + output ~300 tok) × 300건 기준  
> ※ 실제 비용은 thinking 여부, 답변 길이에 따라 변동

### 프로바이더별 사용 분담

| 프로바이더 | 이점 | 담당 모델 |
|----------|------|---------|
| **OpenRouter** | 300+ 모델 통합 API | GPT/Claude/Gemini/Grok/Qwen3.5/MiniMax/Kimi/Cohere/Perplexity/NVIDIA/Solar3 등 |
| **Friendli.ai** | K-EXAONE 독점, 한국 모델 최적화, Prompt caching | K-EXAONE 236B, Qwen3-235B, DeepSeek V3.2, Llama 3.3/3.1 |

**LangChain 연동:**
- OpenRouter: `langchain-openrouter` (공식) — `ChatOpenRouter`
- Friendli.ai: `ChatOpenAI(base_url="https://api.friendli.ai/serverless/v1", api_key=...)` — OpenAI 호환

---

## 4. 한국 회사 모델 현황

| 회사 | 모델 | 크기 | 오픈소스 | GGUF | 우리 보유 | 비고 |
|------|------|------|---------|------|---------|------|
| **LG AI** | K-EXAONE 236B-A23B | 236B MoE | O | O (IQ4 128GB) | X (메모리 초과) | 한국어 세계 10위 |
| **LG AI** | EXAONE 4.5-33B | 33B | O | O (Q4 20GB) | 🔄 다운 중 (AI-395) | 멀티모달 |
| **LG AI** | EXAONE 4.0-32B | 32B | O | O | ✅ Spark (Ollama) | reasoning 모드 |
| **LG AI** | EXAONE 3.5-32B/7.8B | 32/7.8B | O | O | ✅ Spark (Ollama) | |
| **SKT** | A.X-K1 | 519B MoE (33B active) | O | 미확인 | X (너무 큼) | 5개국어 |
| **KT** | Mi:dm 2.0 Base | 11.5B | O | O (Q4 7GB) | 🔄 다운 중 (AI-395) | DuS |
| **KT** | Mi:dm 2.0 Mini | 2.3B | O | O | X | 온디바이스 |
| **KT** | Mi:dm K 2.5 Pro | 32B | O | 미확인 | X | reasoning 특화 |
| **업스테이지** | Solar Pro 2 (v1) | 31B | O | Ollama 있음 | 🔄 다운 중 (Spark) | Ko-MMLU 상위 |
| **업스테이지** | Solar Pro 3 | API only | X | - | OpenRouter | $0.15/$0.60 |
| **네이버** | HyperCLOVAX-SEED-Think-32B | 32B | O | X (GGUF 없음) | X | 한국어 reasoning |
| **네이버** | HyperCLOVAX-SEED-Text-Instruct | 0.5/1.5B | O | X | X | 소형 |
| **네이버** | HyperCLOVAX-SEED-Omni-8B | 8B | O | X | X | 멀티모달 |
| **NCSOFT** | VARCO-8B | 8B | O | 가능 | X | Llama 기반 |

---

## 5. 실험 구조 요약

### 실험 A: 임베딩 비교 (진행 중)

```
21 임베딩 × N LLM(nothink/think) × 300 질문
결과: results/phase5_exp_a_embed/
```

| LLM | 서버 | nothink | think |
|-----|------|---------|-------|
| qwen3.5-27b (Q4_K_M) | AI-395 | ✅ 완료 | 🔄 진행 |
| qwen3.5-35b-a3b (IQ4_XS) | AI-395 | ✅ 완료 | ⏳ 대기 |
| qwen3.5:122b-a10b (Q4_K_M) | Spark | ✅ 완료 | - |
| deepseek-r1:70b | Spark | 🔄 진행 | - |
| exaone3.5:32b | Spark | 🔄 진행 | - |
| gpt-oss:120b | Spark | ⏳ 대기 | - |

### 실험 B: LLM 비교 (준비 중)

```
1 임베딩(gemma-embed-300m, MRR 1위) × 모든 LLM × 300 질문
결과: results/phase5_exp_b_llm/
```

- 로컬 LLM: ~20개 (AI-395 + Spark)
- 프로바이더 LLM: ~11개 (OpenRouter, ~$95)
- 총 ~30+ LLM

### 평가 (RAGAS, 준비 완료)

```
21 임베딩 × 300 질문 = 6,300 평가
Judge: gpt-5.4 (~$142) 또는 gpt-5.4-mini (~$20)
스크립트: scripts/evaluate_ragas.py (dry-run 가능)
```

---

## 6. 인프라 요약

| 서버 | IP | 역할 | 사양 | 스토리지 |
|------|-----|------|------|---------|
| AI-395 | 192.168.50.245 | 임베딩 + LLM (llama.cpp) | MI100 96GB VRAM | 468GB (164GB 여유) |
| DGX Spark | 192.168.50.251 | LLM (Ollama) | GB10 128GB unified | 3.7TB (2.6TB 여유) |
| T7910 | 192.168.50.250 | 벡터스토어 | Dual Xeon 72T, 128GB RAM | 915GB |
| Mac Mini | 192.168.50.241 | 스크립트 실행 | M2 16GB | - |
