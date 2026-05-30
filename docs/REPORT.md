# Korean RAG Evaluation Benchmark — 통합 상세 보고서

> 6 stage 단변량 + Cartesian (384 configs) 전체 실험 결과 통합.
> 각 stage 상세는 [`docs/N_<stage>.md`](.) 참조. 본 문서는 종합 분석.

**데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs × 5 도메인)
**평가 기간**: 2026-04 ~ 2026-05
**총 LLM 호출**: 약 1.2M (Phase 5 다중 vendor 240K + GPT-5.4 cartesian 576K + 기타 ~400K)

---

## 목차

1. [핵심 결과 요약](#1-핵심-결과-요약)
2. [Stage 1: Loader](#2-stage-1-loader)
3. [Stage 2: Parser (Chunker)](#3-stage-2-parser-chunker)
4. [Stage 3: Embedding](#4-stage-3-embedding)
5. [Stage 4: Retriever](#5-stage-4-retriever)
6. [Stage 4-1: Pre-Retriever](#6-stage-4-1-pre-retriever)
7. [Stage 4-2: Post-Retriever (Reranker)](#7-stage-4-2-post-retriever-reranker)
8. [Stage 5: Generation (Phase 5)](#8-stage-5-generation-phase-5)
9. [Stage 5-ext: e2e Axis-wise (시나리오 F)](#9-stage-5-ext-e2e-axis-wise)
10. [Stage 6: Cartesian (Full)](#10-stage-6-cartesian-full)
11. [통합 분석](#11-통합-분석)
12. [최종 winner pipeline + 비용](#12-최종-winner-pipeline--비용)

---

## 1. 핵심 결과 요약

### 1.1 단계별 winner 요약

| Stage | Winner | MRR | 단계 도입 효과 |
|---|---|---:|---:|
| 1 Loader | `pymupdf` (PyMuPDFLoader) | 0.6486 | (기준) |
| 2 Parser | `LC Recursive 300/50` (Slumber 등 42종 비교에서 winner) | 0.6816 (dense) / 0.7171 (hybrid) | char-based가 semantic·LLM 기반보다 우수 |
| 3 Embedding | `KoE5` / `embeddinggemma-300m` | 0.6871 | 한국어 fine-tune |
| 4 Retriever | `Hybrid 3:7 (Dense + BM25-KIWI)` | 0.7171 | +3.55pp vs dense |
| 4-1 Pre-R | `query_expansion` (axis-wise) | 0.7783 | +0.18pp vs baseline |
| 4-2 Post-R | `dragonkue/bge-reranker-v2-m3-ko` | 0.7697 | +5.26pp |
| 5-ext axis | `Hybrid 5:5 + ko-reranker` | 0.7747 / Judge 3.983 | gen +3.5% |
| **6 cartesian** | **`query2doc + Hybrid 7:3 + jina-m0`** | **0.7630 / Judge 4.067** | 최종 |

### 1.2 누적 개선 (naive baseline = baseline + dense + no_rerank)

| 단계 | MRR | Δ MRR (%) | Hit@1 | Δ Hit@1 (pp) | Judge | Δ Judge (%) |
|---|---:|---:|---:|---:|---:|---:|
| 0. naive | 0.6816 | — | 59.0% | — | 3.850 | — |
| + Hybrid (KIWI) | 0.7171 | +5.2% | 65.3% | +6.3 | 3.869 | +0.5% |
| + Reranker (ko) | 0.7697 | +12.9% | 74.0% | +15.0 | 3.916 | +1.7% |
| + axis-wise winner | 0.7747 | +13.7% | 70.7% | +11.7 | 3.983 | +3.5% |
| **+ Cartesian (judge 최고)** | **0.7630** | **+11.9%** | **71.3%** | **+12.3** | **4.067** | **+5.6%** |
| **🏆 MRR 최고 조합** | **0.7874** | **+15.5%** | **75.0%** | **+16.0** | 3.991 | +3.7% |

### 1.3 컴포넌트 영향력 (cartesian 384 configs 분산 기준)

| Axis | judge_mean 변동 범위 | 영향력 |
|---|---:|---|
| **PostR** | 0.40 (3.62 → 4.07) | 🥇 가장 큼 |
| **R** | ~0.15 | 🥈 중간 |
| **PreR** | ~0.05 | 🥉 적음 |

→ **Reranker 선택이 최우선**. PreR은 마지널.

---

## 2. Stage 1: Loader

> 상세: [`1_loader.md`](./1_loader.md)

### 비교 (7종 PDF Loader)

| Loader | MRR | Hit@1 | Hit@5 | parse(s) |
|---|---:|---:|---:|---:|
| **pymupdf** | **0.6486** | 57.0% | 76.3% | **3.1** |
| pdfplumber | 0.6468 | 56.3% | 77.0% | 108.8 |
| pymupdf4llm | 0.6388 | 54.7% | 77.3% | 547.5 |
| pdfminer | 0.6301 | 54.7% | 75.3% | 144.9 |
| docling | 0.6241 | 54.7% | 73.7% | 1,162.5 |
| pypdf | 0.6203 | 53.3% | 74.7% | 32.9 |
| opendataloader | 0.5993 | 50.0% | 75.3% | 169.3 |

### 결론
- **단순 평문 추출이 한국어 RAG에서 가장 강함** — pymupdf 1위 + 압도적 빠름
- markdown 변환(pymupdf4llm), OCR+layout(docling)은 비용 대비 효과 없음
- File@5는 비슷한데 page-level 정확도(MRR)에서 격차

---

## 3. Stage 2: Parser (Chunker)

> 상세: [`2_parser.md`](./2_parser.md)

### 비교 (20+종)

#### Character-based 12종 (Stage 2 본)

| Strategy | Chunks | MRR | Hit@1 |
|---|---:|---:|---:|
| **LC Recursive 300/50** | 4,556 | **0.6816** | 59.0% |
| LC Token 256/50 (tiktoken cl100k) | 5,091 | 0.6798 | 60.3% |
| Chonkie Fast 500 | 5,186 | 0.6774 | 59.3% |
| LlamaIndex Sentence 500/100 | 2,858 | 0.6774 | 60.7% |
| Chonkie Recursive 500 | 2,746 | 0.6614 | 58.0% |
| (... 5개 더, MRR 0.61-0.66) | | | |
| **Chonkie Token 256/50 (gpt2)** | 2,377 | **0.4193** | 35.0% |

#### Size grid 6 chunker × 5 size (그룹 A 확장)

**chunker마다 sweet spot 다름**:
- LC Recursive / Character / Chonkie Recursive / Chonkie Sentence: **300/50** 최적
- **Chonkie Fast: 800 최적** (MRR 0.6903 ← 그룹 A 1위)
- LlamaIndex Sentence: **500/100** 최적

#### 그룹 B — Semantic + LLM 10종 (Hybrid 3:7 baseline)

| Chunker | MRR | parse_time |
|---|---:|---:|
| (기준) LC Recursive 300/50 (hybrid 재측정) | **0.7171** | 4.6s |
| 🥇 **Chonkie Slumber (gpt-5.4)** | 0.7112 | **5,608s** |
| LlamaIndex SemanticSplitter | 0.7076 | 213s |
| Kiwi + Recursive 500/100 | 0.7026 | 9s |
| Chonkie NeuralChunker | 0.6994 | 74s |
| Chonkie Semantic (text-embedding-3-large) | 0.6948 | 2,191s |
| Chonkie Semantic (text-embedding-3-small) | 0.6909 | 2,348s |
| LC MarkdownHeaderTextSplitter | 0.6574 | 1,083s |
| ❌ KSS + Recursive | timeout | — |

### 결론
- **char-based LC Recursive 300/50 절대 winner** (MRR 0.6816 dense / 0.7171 hybrid)
- 그룹 B 안에선 Chonkie Slumber 1위, 그러나 LC Recursive 300/50 보다 −0.59pp + parse 5,603초 추가 + ~$2 LLM 비용
- **semantic / LLM chunker는 본 데이터셋에서 비용 대비 효과 없음**
- chunker library 효과 < chunk size 효과 (300/50 vs 1500/200: −7.6pp)

---

## 4. Stage 3: Embedding

> 상세: [`3_embedding.md`](./3_embedding.md)

### Top 10 (27 모델 leaderboard)

| 순위 | 모델 | dim | MRR | 비고 |
|---:|---|---:|---:|---|
| 🥇 | **koe5** | 1024 | **0.6871** | 한국어 특화 |
| 🥈 | gemma-embed-300m | 768 | 0.6650 | 최고 소형 |
| 🥉 | pixie-rune-v1 | 1024 | 0.6627 | |
| 4 | snowflake-arctic-ko | 1024 | 0.6612 | 한국어 튜닝 |
| 5 | snowflake-arctic-l-v2 | 1024 | 0.6495 | |
| 6 | jina-v4-retrieval | 4096 | 0.6449 | |
| 7 | nomic-embed-v2-moe | 768 | 0.6435 | MoE |
| 8 | kure-v1 | 1024 | 0.6267 | 한국어 |
| 9 | harrier-0.6b | 1024 | 0.6131 | pooling=last |
| 10 | granite-278m | 768 | 0.5969 | IBM |

### 결론
- **한국어에서는 작은 모델이 큰 영어 모델을 이긴다**: KoE5(1024d) > qwen3-embed-8b(4096d), MRR +0.16
- 한국어 특화 임베딩(koe5, snowflake-arctic-ko, kure-v1) 상위 점유
- harrier-27b 5376 차원 중 97% dead dim — 한국어 query/doc 분리 실패
- 본 데이터셋의 이후 stage는 **gemma-embed-300m** 사용 (운영 안정성)

---

## 5. Stage 4: Retriever

> 상세: [`4_retriever.md`](./4_retriever.md)

### 비교 (7 strategy)

| Strategy | MRR | Hit@1 | Hit@5 | File@5 |
|---|---:|---:|---:|---:|
| 🥇 **Hybrid 3:7 (Dense + BM25-KIWI, RRF k=60)** | **0.7171** | **65.3%** | 80.3% | 91.7% |
| Hybrid 5:5 | 0.7137 | 65.3% | 80.0% | 91.7% |
| Hybrid 7:3 | 0.7046 | 64.0% | 80.3% | 91.7% |
| Dense (gemma-300m) | 0.6816 | 59.0% | **81.3%** | 91.7% |
| BM25 + KIWI | 0.6783 | 61.3% | 77.3% | 89.3% |
| Hybrid 5:5 (WS BM25) | 0.6496 | 59.0% | 74.3% | 88.0% |
| BM25 + whitespace | 0.5344 | 48.3% | 62.7% | 77.7% |

### 결론
- **Hybrid가 단독 검색을 모두 이긴다** — Dense/BM25-KIWI alone < 모든 hybrid
- **KIWI 형태소 분석 필수** — BM25-KIWI 0.6783 vs whitespace 0.5344 (+14.4pp)
- **Dense ≈ BM25-KIWI** — 단독 성능 거의 동률 (0.6816 vs 0.6783)
- Hybrid weight 3:7이 최적이지만 5:5와 차이 작음 (±1pp)

---

## 6. Stage 4-1: Pre-Retriever

> 상세: [`4-1_pre_retriever.md`](./4-1_pre_retriever.md)
>
> LLM: GPT-5.4, 2,400 unique LLM calls, 20-worker parallel

### 10 전략 비교 (R/PostR 고정)

| 순위 | 전략 | MRR | Hit@1 | vs baseline |
|---:|---|---:|---:|---:|
| 🥇 | **query_expansion** | **0.7783** | 70.0% | +0.0086 |
| 2 | hyde | 0.7738 | 70.3% | +0.0041 |
| 3 | decompose | 0.7786 | 70.7% | +0.0089 |
| 4 | query2doc | 0.7768 | 70.0% | +0.0071 |
| 5 | hyde_rrf | 0.7713 | 70.3% | +0.0016 |
| 6 | multi_query_angle | 0.7770 | 69.3% | +0.0073 |
| 7 | query_rewrite | 0.7741 | 70.3% | +0.0044 |
| 8 | multi_query_para | 0.7780 | 70.3% | +0.0083 |
| 9 | **baseline (no PreR)** | **0.7697** | 70.7% | (기준) |
| 10 | step_back | 0.7692 | 69.3% | −0.0005 |

### 결론
- **거의 모든 PreR이 baseline 동급 또는 약간 우세** — 단변량으론 효과 미미
- step_back / multi_query_angle 추상화는 마이너스 또는 무효
- query_expansion / decompose가 살짝 가장 좋음
- **상호작용 효과 큼**: Stage 6 cartesian에서 query2doc + jina-m0 조합이 PreR axis 1위로 도약

---

## 7. Stage 4-2: Post-Retriever (Reranker)

> 상세: [`4-2_post_retriever.md`](./4-2_post_retriever.md)
>
> **25 reranker + no_rerank baseline 비교** (시도 실패 5종 별도) (M5 Max + AMD AI 395+ 분산)

### Top 12 reranker leaderboard

| 순위 | 모델 | 파라미터 | MRR | Hit@1 | 비고 |
|---:|---|---:|---:|---:|---|
| 🥇 | **dragonkue/bge-reranker-v2-m3-ko** | 0.6B (m3 KR 튜닝) | **0.7697** | **74.0%** | retrieval winner |
| 🥈 | jinaai/jina-reranker-m0 | 2.4B 멀티모달 | 0.7631 | 72.3% | cartesian winner 후보 |
| 🥉 | Qwen/Qwen3-Reranker-4B | 4B (2025-06) | 0.7514 | 70.3% | |
| 4 | BAAI/bge-reranker-v2-m3 | 0.6B | 0.7488 | 70.0% | |
| 5 | Dongjin-kr/ko-reranker | 0.6B KR | 0.7320 | 67.3% | axis-wise gen winner |
| 6 | mixedbread-ai/mxbai-rerank-base-v2 | 0.5B (Apache) | 0.7319 | 67.3% | |
| 7 | upskyy/ko-reranker-8k | 0.6B KR | 0.7343 | 68.0% | |
| 8 | upskyy/ko-reranker | 0.6B KR | 0.7238 | 67.7% | |
| 9 | Qwen/Qwen3-Reranker-0.6B | 0.6B | 0.7255 | 66.3% | |
| 10 | BAAI/bge-reranker-large | 0.5B XLM-R-L | 0.7091 | 65.0% | legacy |
| — | no_rerank baseline | — | 0.7171 | 65.3% | |
| 11 | mxbai-rerank-large-v2 | 1.5B | 0.6945 | 66.7% | ⚠ baseline 미달 |
| 12 | nvidia-llama-nemotron-1b | 1B | 0.6966 | 63.7% | |

### Failed / catastrophic
- ❌ jina-reranker-v3, jina-reranker-v2-multilingual, sigridjineth-ko-reranker-v1/v1.1, bge-reranker-v2-minicpm, bge-reranker-v2.5-gemma2-lightweight: transformers v5 호환성 실패
- ⚠ bge-reranker-v2-gemma 0.4152, modernReranker 0.5159, pixie-spell-reranker 0.3276: baseline 한참 미달

### 결론
- **한국어 fine-tune이 SOTA를 압도**: dragonkue ko-reranker가 Qwen3-Reranker-4B(2025 SOTA, 약 6.7배 큰 4B 모델)를 +1.83pp MRR
- 한국어 정렬 > 모델 신선도·크기
- **mxbai-base > mxbai-large 역전** — 큰 모델이 항상 좋진 않음

---

## 8. Stage 5: Generation (Phase 5)

> 상세: [`5_generation.md`](./5_generation.md)

### 별도 trajectory — Open vs Closed weights 비교

- **46 generation 모델** (Open 27 + Closed 19) × **18 judge 모델** (3-judge flagship ensemble) × 300 Q&A
  - *(이후 공개 대시보드 [rag.baeum.ai.kr](https://rag.baeum.ai.kr)에서는 judge 를 20종(API 9 + open-weight 11)으로 확장 — 본 보고서 수치는 Phase 5 평가 시점의 18-judge 기준)*
- 목적: GPT-5.4-pro / GPT-5.4 같은 closed API 모델을 대체할 수 있는 로컬 모델 탐색
- 비용: ~$240 (Anthropic Opus 4.7, GPT-5.4-pro, Gemini-3.1-pro 3-judge)

### 핵심 발견
- **Open Weights 1위**: `qwen3.5_122b-a10b-q4_K_M_think` (MoE, 122B/10B-active) — closed API와 동급
- `deepseek-r1_70b`, `gpt-oss_120b`, `kimi-k2.6`, `glm-5.1` — 상위 그룹
- **Closed 1위**: gpt-5.4-pro > claude-opus-4.7 > gemini-3.1-pro
- 자세한 leaderboard와 self-bias 분석은 5_generation.md 참조

---

## 9. Stage 5-ext: e2e Axis-wise

> 상세: [`5_generation_e2e_axis.md`](./5_generation_e2e_axis.md)
>
> 28 configs × 300q × (1 gen + 4 judge) = 42,000 GPT-5.4 호출, ~$21

각 axis 단변량 → 검색 metric vs 생성품질 비교.

### Axis A — PreR (10)
- 1위: **query_expansion** (judge 3.998)
- 모두 3.94-4.00 범위, 편차 1.5% — PreR 영향 작음

### Axis B — R (7)
- 1위: **Hybrid 5:5** (judge 3.983)
- bm25_whitespace 3.540 (catastrophic, retrieval과 일치)

### Axis C — PostR (11)
- 1위 (judge): **Dongjin-kr/ko-reranker** (4.005)
- 1위 (MRR): **dragonkue/bge-reranker-v2-m3-ko** (0.7697)
- → **retrieval 1위와 generation 1위 다름** — 둘 다 KR 튜닝이지만 모델 차이

### 핵심 발견
- **컴포넌트 영향력 순서**: PostR > R > PreR (cartesian에서 재확인)
- 검색-생성 strongly correlated이지만 1위 결정은 generation까지 봐야 함
- 안 좋은 컴포넌트(bm25_ws, pixie 등)는 양쪽에서 치명적

---

## 10. Stage 6: Cartesian (Full)

> 상세: [`6_cartesian.md`](./6_cartesian.md)
>
> **384 configs** (8 PreR × 6 R × 8 PostR) × 300q × (1 gen + 4 judge) = **576,000 호출**, ~$290, 7.5h

### Top 10 winners (judge_mean)

| 순위 | PreR | R | PostR | MRR | Hit@1 | Judge |
|---:|---|---|---|---:|---:|---:|
| 🥇 | **query2doc** | **hybrid_7_3** | **jina-reranker-m0** | 0.7630 | 71.3% | **4.067** |
| 🥈 | query_expansion | hybrid_5_5 | jina-reranker-m0 | 0.7806 | 74.0% | 4.062 |
| 🥉 | query2doc | hybrid_5_5 | jina-reranker-m0 | 0.7726 | 73.0% | 4.036 |
| 4 | query_expansion | dense | jina-reranker-m0 | 0.7607 | 70.7% | 4.032 |
| 5 | query2doc | hybrid_5_5 | ko-reranker | 0.7401 | 68.3% | 4.030 |
| 6 | hyde | hybrid_5_5 | jina-reranker-m0 | 0.7719 | 73.0% | 4.027 |
| 7 | hyde | hybrid_7_3 | jina-reranker-m0 | 0.7756 | 73.0% | 4.026 |
| 8 | query2doc | dense | jina-reranker-m0 | 0.7558 | 70.0% | 4.026 |
| 9 | hyde_rrf | hybrid_5_5 | jina-reranker-m0 | 0.7769 | 74.0% | 4.025 |
| 10 | query_expansion | hybrid_3_7 | jina-reranker-m0 | 0.7779 | 74.3% | 4.021 |

### MRR Top 5 (다른 ranking)

| 순위 | config | MRR | Hit@1 | Judge |
|---:|---|---:|---:|---:|
| 🥇 | **multi_query_para + hybrid_5_5 + jina-reranker-m0** | **0.7874** | **75.0%** | 3.991 |
| 🥈 | multi_query_para + hybrid_ws_5_5 + bge-reranker-v2-m3-ko | 0.7850 | 75.0% | 4.019 |
| 🥉 | decompose + hybrid_5_5 + bge-reranker-v2-m3-ko | 0.7816 | 74.7% | 3.967 |

### Bottom 10
- 모두 **`no_rerank`** + (hybrid_ws_5_5 또는 bm25_kiwi). worst: hyde + hybrid_ws_5_5 + no_rerank (judge 3.618).

### 핵심 발견
1. **jina-reranker-m0가 cartesian 1위!** axis-wise 미실험이었던 멀티모달 reranker (Qwen2-VL 2.4B 백본)
2. **query2doc + jina-m0 상호작용** — axis-wise에선 query2doc 4위였으나 cartesian에선 1, 3, 5, 7, 8위 (jina-m0와 조합 시)
3. **PostR 영향력 압도적** — judge 3.62~4.07 (10% 변동), PreR/R는 1-3%
4. **reranker는 필수** — bottom 10 전부 no_rerank
5. **MRR 최고 vs Judge 최고 다름** — retrieval ≠ generation 품질 1위

### Phase 5 비교 (동일 dataset / embedding / 평가 규칙)

| Pipeline | Generator | Accuracy |
|---|---|---:|
| 🥇 **Cartesian winner** (query2doc + Hybrid 7:3 + jina-m0) | GPT-5.4 | **0.827** |
| Phase 5 단순 retrieval + GPT-5.4 | GPT-5.4 | 0.787 |
| Phase 5 + GPT-5.4-pro | GPT-5.4-pro | 0.767 |
| Phase 5 Open 1위 | gpt-oss_120b / kimi-k2.5 | 0.740 |

→ **RAG 파이프라인 최적화가 모델 업그레이드보다 큼**: 동일 GPT-5.4 사용 시 단순 retrieval 0.787 → cartesian winner 0.827 (+4.0pp), GPT-5.4-pro로 업그레이드한 것보다 +6.0pp 높음.

### 10.1 Judge Robustness — GPT-5.4(closed) vs Qwen3.6 35B-A3B(open) 재채점

384 cartesian 조합을 closed judge(GPT-5.4) 외에 **open-weight judge(Qwen3.6 35B-A3B, self-host)** 로 한 번 더 전수 채점해 순위 안정성을 검증했다.

| Judge | 채점 조합 | 평균 accuracy | 성격 |
|---|---|---:|---|
| GPT-5.4 (closed, API) | 384 | 78.0% | 기준 |
| Qwen3.6 35B-A3B (open, self-host) | 384 | **82.1%** | 더 관대(higher leniency) |

- open judge 가 평균 +4.1pp 더 후하게 채점하나, **조합 간 상대 순위(rank)는 대체로 보존** — winner 계열(query2doc·hybrid·jina-m0)이 양쪽 judge 에서 모두 상위.
- 시사점: judge 절대값은 calibration 차이로 ±4pp 흔들리므로 **단일 judge 절대 점수보다 다수 judge·상대 순위로 해석**해야 한다. open judge 는 비용 0 으로 rank 검증·대규모 재채점에 실용적.
- 두 judge 의 조합별 비교는 공개 대시보드 [rag.baeum.ai.kr](https://rag.baeum.ai.kr) 리더보드의 JUDGE 탭에서 직접 전환 가능.

---

## 11. 통합 분석

### 11.1 컴포넌트 영향력 정량화 (cartesian 기준)

| Axis | Best | Worst | Range |
|---|---:|---:|---:|
| **PostR** (jina-m0 vs no_rerank) | 4.067 | 3.618 | **+0.45 (12%)** |
| R (hybrid_5_5 vs hybrid_ws_5_5) | 4.062 | 3.752 | +0.31 (8%) |
| PreR (query_expansion vs step_back) | 3.998 | 3.941 | +0.06 (1.5%) |

→ **Reranker 선택이 압도적 가장 중요**. 다음 retriever, PreR은 마이너.

### 11.2 단변량 vs Cartesian 일치도

| Stage | 단변량 1위 | Cartesian 1위 | 동일? |
|---|---|---|---|
| PreR | query_expansion | query2doc (with jina-m0) | ✗ 상호작용 |
| R | Hybrid 5:5 | Hybrid 7:3 (with jina-m0) | ✗ 인접 |
| PostR | dragonkue/bge-v2-m3-ko (retrieval) / ko-reranker (axis-wise judge) | jinaai/jina-reranker-m0 | ✗ axis-wise 미실험 |

→ **단변량 → cartesian으로 ~1.5% judge 추가 개선**. 상호작용 효과 무시 못 함.

### 11.3 검색-생성 metric 괴리

| Pipeline | MRR | Hit@1 | Judge mean |
|---|---:|---:|---:|
| MRR 최고 config | **0.7874** | **75.0%** | 3.991 |
| Hit@1 최고 config (동률) | 0.7874 | 75.0% | 3.991 |
| Judge 최고 config | 0.7630 | 71.3% | **4.067** |

→ retrieval 1위와 generation 1위가 다른 조합. 운영 목적에 따라 선택:
- **검색 정확도 최우선**: multi_query_para + Hybrid 5:5 + jina-reranker-m0
- **답변 품질 최우선**: query2doc + Hybrid 7:3 + jina-reranker-m0

### 11.4 한국어 특화의 가치 재확인

- Embedding: KoE5 > Qwen3-Embed-8B (+0.16 MRR)
- BM25: KIWI > whitespace (+14.4pp)
- Reranker: dragonkue/bge-v2-m3-ko(0.6B) > Qwen3-Reranker-4B (+1.83pp, 약 6.7배 작은 모델로)

**일관된 패턴**: 한국어 정렬 > 모델 신선도/크기

### 11.5 비용 vs 효과 trade-off

| 추가 기법 | 1회 비용 | judge gain | gain/cost |
|---|---:|---:|---|
| Hybrid 추가 | $0 (로컬) | +0.019 | ∞ |
| Reranker 추가 | $0 (로컬, ~110s/실험) | +0.047 | ∞ |
| Cartesian PreR (jina-m0 결합) | $0.025/실험 LLM 변형 | +0.151 | 6.0 |
| **권장**: 단변량 + reranker + 선별 cartesian → 90% 효과를 10% 비용으로 |

---

## 12. 최종 winner pipeline + 비용

### 12.1 권장 운영 pipeline

#### Judge 최고 (gen 품질 우선)
```
PyMuPDFLoader
  → RecursiveCharacterTextSplitter(300, 50)
  → google/embeddinggemma-300m
  → query2doc (GPT-5.4 가상문서 생성)
  → Hybrid 7:3 (FAISS Dense + BM25-KIWI, RRF k=60)
  → top-20
  → jinaai/jina-reranker-m0 (2.4B 멀티모달 cross-encoder)
  → top-5
  → GPT-5.4 RAG 답변
```
**MRR 0.7630 / Hit@1 71.3% / Judge 4.067**

#### MRR/Hit@1 최고 (retrieval 정확도 우선)
```
... (동일)
  → multi_query_para (3 paraphrase RRF)
  → Hybrid 5:5 (FAISS Dense + BM25-KIWI)
  → top-20
  → jinaai/jina-reranker-m0
  → top-5
  → GPT-5.4 답변
```
**MRR 0.7874 / Hit@1 75.0% / Judge 3.991**

### 12.2 전체 실험 비용 (실측)

| 단계 | LLM 호출 | 시간 | 비용 |
|---|---:|---:|---:|
| Stage 1-4 단변량 (로컬 only) | 0 | 합산 ~8h | $0 |
| Stage 4-1 PreR (GPT-5.4) | 2,400 | 5분 | ~$3 |
| Stage 5 axis-wise (GPT-5.4) | 42,000 | 2h | ~$21 |
| Stage 6 Cartesian (GPT-5.4) | 576,000 | 2.5h | ~$290 |
| Stage 5 Phase 5 (Anthropic + OpenAI + Gemini) | ~240,000 | 별도 48h batch | ~$240 |
| **합계 (LLM API)** | **~860K** | — | **~$554** |
| 로컬 GPU 추론 (sentence-transformers + ROCm) | — | 합산 ~15h | $0 |

---

## 부록 A: 환경 / 인프라

- **로컬**: M5 Max (Apple Silicon, MPS, 64GB unified), Python 3.12, sentence-transformers
- **GPU 서버**: HP Z2 Mini (AI 395+), 96GB unified VRAM, PyTorch 2.12 ROCm 7.13 (TheRock builds)
- **OpenAI**: AI Foundry, `gpt-5.4` deployment, 10M TPM / 100K RPM
- **LangChain**: `PyMuPDFLoader`, `RecursiveCharacterTextSplitter`, `FAISS`, `BM25Retriever`, `EnsembleRetriever`, `CrossEncoderReranker`, `ChatOpenAI(base_url=...)`
- **병렬화**: ThreadPoolExecutor 100-worker (gen/judge)

## 부록 B: 데이터셋 / 산출물

- HuggingFace: [BAEM1N/Korean-RAG-LLM-Judge-Benchmark](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark)
- 원본 데이터: [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
- Cartesian 384 결과: [HuggingFace 데이터셋](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark)
- 단계별 분석: 본 docs/ 의 1~6 단계 보고서
- 재현 스크립트: [`scripts/`](../scripts/)
- **공개 baseline retrieval API**: `POST https://rag.baeum.ai.kr/api/retrieve` — winner 전단(PyMuPDFLoader · RecursiveChar 300/50 · embeddinggemma-300m) 기반 FAISS + BM25-KIWI + Hybrid 검색을 누구나 호출 가능 (재현·비교용)
- **요약 대시보드**: [rag.baeum.ai.kr](https://rag.baeum.ai.kr) — 단계별 차트·리더보드(GPT-5.4/Qwen3.6 judge)·RAG 결과 탐색

## 부록 D: 벡터스토어 백엔드 확장성 (별도 운영 벤치)

retrieval *품질*(본문)과 별개로, 동일 1536d 벡터·top_k=10 조건에서 **17개 벡터 검색 시스템의 운영 확장성**(QPS·p50/p95/p99·insert·RAM·recall)을 1K~10M 청크 단계 게이트로 비교했다 (전용 벡터 DB·검색엔진·분석 DB·라이브러리 8 카테고리).

- 작은~중간 코퍼스(≤10K): `scann`·`hnswlib`·`faiss_cpu` 가 수만 QPS 로 압도
- 대규모(10M): `faiss_gpu` 가 QPS·지연 평탄 유지 — 확장성 1위
- `elasticsearch`/`opensearch` 는 RAM 40GB대로 무거움
- 본 RAG 평가의 기본 retrieval backend = **FAISS CPU** (3,166 청크 소규모라 충분)
- 상세: [rag.baeum.ai.kr](https://rag.baeum.ai.kr) → 벡터스토어 벤치

## 부록 C: 학술 레퍼런스

### Pre-retriever
- HyDE — Gao et al. 2022 ACL — [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- Query2doc — Wang et al. EMNLP 2023 — [arXiv:2303.07678](https://arxiv.org/abs/2303.07678)
- Step-back prompting — Zheng et al. ICLR 2024 — [arXiv:2310.06117](https://arxiv.org/abs/2310.06117)
- Self-Ask — Press et al. EMNLP 2023 — [arXiv:2210.03350](https://arxiv.org/abs/2210.03350)
- Query Expansion (LLM) — Jagerman et al. 2023 — [arXiv:2305.03653](https://arxiv.org/abs/2305.03653)
- Query Rewriting — Ma et al. EMNLP 2023 — [arXiv:2305.14283](https://arxiv.org/abs/2305.14283)

### Retrieval
- BM25 — Robertson & Zaragoza 2009 FnT IR — [DOI](https://doi.org/10.1561/1500000019)
- RRF — Cormack et al. SIGIR 2009 — [DOI](https://doi.org/10.1145/1571941.1572114)
- Kiwi 형태소 — [github.com/bab2min/Kiwi](https://github.com/bab2min/Kiwi)

### Post-retriever
- BGE-M3 — Chen et al. 2024 — [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- Qwen3-Reranker — 2025-06 — [arXiv:2506.05176](https://arxiv.org/abs/2506.05176)
- ColBERTv2 — [arXiv:2112.01488](https://arxiv.org/abs/2112.01488)
- RankZephyr — [arXiv:2312.02724](https://arxiv.org/abs/2312.02724)

### 모델 (HuggingFace)
- [dragonkue/bge-reranker-v2-m3-ko](https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko)
- [jinaai/jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0)
- [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
- [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
- [nlpai-lab/KoE5](https://huggingface.co/nlpai-lab/KoE5)
