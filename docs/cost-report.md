# RAG Benchmark — API 비용 보고서

**작성일**: 2026-04-20
**대상**: Phase 4 임베딩 확장 + Phase 5 Exp B 생성 확장 + Flagship 3-Judge 판정

---

## 1. 입력 데이터 규모

| 항목 | 값 |
|---|---|
| 질문 수 | 300 |
| 청크 수 | 3,166 |
| 도메인 | 5 (finance, public, medical, law, commerce) |
| 고정 임베딩 (생성/판정용) | gemma-embed-300m (Phase 4 2위) |
| 평균 생성 input | 923 토큰 (min 376 / max 2,991) |
| 평균 생성 output | 100 토큰 (3문장 한국어) |
| 평균 Judge input | 385 토큰 (static 255 + candidate 100) |
| Judge output | 1 토큰 |

토큰 계산 기준: 한국어 평균 2.5 chars/token.

---

## 2. 생성 LLM 리스트 — 27개 config

### 2.1 OpenAI Direct API (6개)

| 모델 | Batch $/1M (in/out) | 300 Q&A 비용 |
|---|---|---|
| gpt-5.4-pro (thinking) | $15 / $90 | **$6.85** |
| gpt-5.4 (thinking) | $1.25 / $7.50 | $0.58 |
| gpt-5.4-mini | $0.125 / $1.00 | $0.07 |
| gpt-5.4-nano | $0.025 / $0.20 | $0.01 |
| gpt-5.3-chat (nothink) | ~$1.00 / $5.00 | $0.42 |
| o4-mini (reasoning) | $0.55 / $2.20 | $0.22 |
| **소계** | | **$8.15** |

### 2.2 Anthropic Direct API (4개)

| 모델 | Batch $/1M | 300 Q&A |
|---|---|---|
| claude-opus-4.7 | $2.50 / $12.50 | **$1.07** |
| claude-sonnet-4.6 | $1.50 / $7.50 | $0.64 |
| claude-sonnet-4.5 | $1.50 / $7.50 | $0.64 |
| claude-haiku-4.5 | $0.50 / $2.50 | $0.22 |
| **소계** | | **$2.57** |

### 2.3 Google Direct API (4개)

| 모델 | Batch $/1M | 300 Q&A |
|---|---|---|
| gemini-3.1-pro | $1.00 / $6.00 | **$0.46** |
| gemini-3.1-flash | $0.25 / $1.50 | $0.11 |
| gemini-3.1-flash-lite | $0.125 / $0.75 | $0.06 |
| gemini-3-pro (legacy) | ~$0.625 / $5 | $0.33 |
| **소계** | | **$0.96** |

### 2.4 Thinking 변형 (4개, 출력 3배 가정)

| 모델 | 300 Q&A |
|---|---|
| gpt-5.4 (think) | ~$1.00 |
| claude-opus-4.7 (think) | ~$3.20 |
| gemini-3.1-pro (think) | ~$1.40 |
| claude-sonnet-4.6 (think) | ~$1.90 |
| **소계** | **$7.50** |

### 2.5 OpenRouter (9개, 3사 제외 + 로컬 불가)

Batch API 미지원 → 표준 가격. 캐싱 일부 가능.

| 모델 | 단가 추정 | 300 Q&A |
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
| **소계** | | **~$6.85** |

### 생성 총합

| 구분 | 비용 |
|---|---|
| Direct API (18 configs) | $19.18 |
| OpenRouter (9 configs) | $6.85 |
| **생성 총합 (27 configs)** | **~$26** |

---

## 3. 판정(Judge) — Flagship 3사

**Judge 대상**: 12 local + 27 provider = **39 generation configs**
**Judge 호출**: 4 metric × 300 Q&A × 39 LLMs = **46,800 calls per judge**
**Judge input**: 385 토큰/call × 46,800 = **18.0M tokens per judge**

### 3.1 Prompt Caching 적용 전 (Batch only)

| Judge | Batch $/1M | Input 비용 | Output 비용 | 소계 |
|---|---|---|---|---|
| **gpt-5.4-pro** | $15 / $90 | $270 | $4.2 | **$274** |
| **claude-opus-4.7** | $2.50 / $12.50 | $45 | $0.6 | **$46** |
| **gemini-3.1-pro** | $1 / $6 | $18 | $0.3 | **$18** |
| **3-judge 총합** | | | | **~$338** |

### 3.2 Prompt Caching 적용 후

Static 부분 (template + question + target) = 255/385 = **66% 캐시 가능**.
같은 질문에 대해 27 LLM의 candidate만 바뀜 → 매우 cache-friendly.

| Judge | 캐시 할인율 | 절감 후 |
|---|---|---|
| **gpt-5.4-pro** (OpenAI auto cache) | 50% off cached | **~$150** |
| **claude-opus-4.7** (explicit cache 0.1x) | 90% off cached | **~$15** |
| **gemini-3.1-pro** (context cache 0.2x) | 80% off cached | **~$8** |
| **3-judge 캐싱 총합** | | **~$173** |

### 판정 절감 요약

- Batch only: $338
- Batch + Caching: **$173** (−50%)

---

## 4. 임베딩 API 추가 (Phase 4 확장)

총 텍스트: 3,166 청크 + 300 질문 = **3,466 texts**, 평균 200 토큰 ≈ **700K 토큰**

| 모델 | dim | 가격 $/1M | 우리 비용 |
|---|---|---|---|
| OpenAI text-embedding-3-small | 1536 | $0.02 | **$0.015** |
| OpenAI text-embedding-3-large | 3072 | $0.13 | **$0.09** |
| Google gemini-embedding-001 | 768/1536/3072 | $0.025 | **$0.018** |
| Google text-multilingual-embedding-002 | 768 | $0.025 | **$0.018** |
| **임베딩 총합** | | | **$0.14** |

현재 Phase 4의 **27 local + API 4 = 31 임베딩**로 확장.

---

## 5. 예산 상한 & 시나리오

### 5.1 풀 스펙 (권장)

| 항목 | 비용 |
|---|---|
| 생성 27 configs | $26 |
| 임베딩 API 4 | $0.14 |
| 판정 3 flagship (batch + caching) | $173 |
| **총합** | **~$200** |

### 5.1.1 Provider별 예산 분해 (풀 스펙)

| Provider | 생성 | 임베딩 | 판정 | **Provider 소계** |
|---|---:|---:|---:|---:|
| **OpenAI** | $9.15 (gpt-5.4-pro/5.4/mini/nano, gpt-5.3, o4-mini, +gpt-5.4 think) | $0.10 (3-small + 3-large) | **$150** (gpt-5.4-pro judge, auto cache) | **~$160** |
| **Anthropic** | $7.67 (opus-4.7, sonnet-4.6/4.5, haiku-4.5, +2 think variants) | — | **$15** (opus-4.7 judge, explicit cache 0.1x) | **~$23** |
| **Google** | $2.36 (gemini-3.1-pro/flash/flash-lite, 3-pro, +pro think) | $0.04 (gemini-embedding-001 + text-multilingual-002) | **$8** (gemini-3.1-pro judge, context cache 0.2x) | **~$10** |
| **OpenRouter** | $6.85 (grok, kimi, minimax, qwen3-max/3.6-plus, glm-5.1, sonar, command-a, mistral-large-3) | — | — | **~$7** |
| **합계** | **$25.95** | **$0.14** | **$173** | **~$200** |

#### Provider별 특징

- **OpenAI 비중 80%**: GPT-5.4-Pro가 단독 judge 비용의 80%를 차지 (입력 $15/1M batch). 캐싱 후에도 가장 비쌈.
- **Anthropic 효율 최고**: Claude Opus 4.7의 explicit cache 0.1x 할인 → judge 비용 90% 절감. 비용 대비 최고의 flagship judge.
- **Google 최저가**: Gemini 3.1 Pro가 batch + context caching 스택으로 judge $8. 거의 "덤" 수준.
- **OpenRouter 보완**: 3사가 제공 안 하는 flagship들 (Grok, Kimi-K2.5 1T, MiniMax-M2.7 230B 등)만 선택적으로 사용 → $7 수준.

#### 만약 GPT-5.4-Pro 제외 (옵션 A)

| Provider | Provider 소계 |
|---|---:|
| OpenAI (Pro 대신 standard GPT-5.4 judge) | ~$50 |
| Anthropic | ~$23 |
| Google | ~$10 |
| OpenRouter | ~$7 |
| **합계** | **~$90** (50% 절감) |

### 5.2 예산 절감 시나리오

**옵션 A — Pro 제외**: GPT-5.4-Pro 대신 GPT-5.4 standard 사용
- Judge 비용: $173 → **~$75** (GPT-5.4 Batch $1.25/$7.50)
- 총합: **~$101**
- 품질: 약간 하락 (Pro vs Standard benchmark 차이 2-5% 수준)

**옵션 B — OpenRouter 제외**: 생성 Direct 18개만
- 생성: $26 → $19.18
- 총합: **~$193**

**옵션 C — 최저가**: Pro 제외 + OpenRouter 제외
- 총합: **~$95**

### 5.3 예산 계획 (권장 배분)

| 단계 | 예산 |
|---|---|
| 버퍼 | +20% 재실행/오류 대비 |
| **총 상한** | **~$250** (풀 스펙 + 버퍼) |

---

## 6. 실행 타임라인

| 단계 | 소요 시간 | 비고 |
|---|---|---|
| Direct API 배치 제출 | 24h 대기 | OpenAI/Claude 비동기 |
| OpenRouter 실시간 | ~3-5시간 | 9 LLM × 300 Q&A 순차 |
| 임베딩 API 호출 | ~30분 | 4 모델 × 3,466 texts |
| Judge batch 제출 | 24-48h | 3 flagship × 캐싱 |
| **총 소요** | **~48-72시간** | |

---

## 7. API 키 요구사항

| Provider | 환경변수 | 용도 |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | 생성 6 + 임베딩 2 + Judge 1 |
| Anthropic | `ANTHROPIC_API_KEY` | 생성 4 + Judge 1 |
| Google | `GOOGLE_API_KEY` 또는 Vertex SA JSON | 생성 4 + 임베딩 2 + Judge 1 |
| OpenRouter | `OPENROUTER_API_KEY` | 생성 9 (3사 제외) |

---

## 8. 이 문서 사용법

1. API 키 확보 후 이 문서의 리스트 기준으로 스크립트 실행
2. 실제 비용은 각 provider 대시보드에서 실측 — 이 문서는 예상치
3. Exchange rate: 1 USD ≈ 1,350 KRW 기준 (2026-04 평균)
4. 최악 시나리오(캐싱 hit rate 0% + Pro full): ~$340
5. 이상 시나리오(캐싱 hit rate 90%): ~$150

## 9. 리스크

- **Thinking 모드 output 폭증**: 일부 reasoning 모델이 2000+ 토큰 생성 → 비용 5-10배 증가 가능. `num_predict` / `max_tokens` 상한 500-1000으로 제한 권장
- **Batch 실패 재제출**: Provider 측 오류 시 재제출 필요 → 20% 버퍼 확보
- **OpenRouter markup**: 일부 route에 5-10% 추가 마크업 → 예상보다 약간 높을 수 있음
- **Caching TTL 만료**: Anthropic 5분 / OpenAI 10분 — batch job 내 질문 순서 조정으로 hit rate 최대화 필요
