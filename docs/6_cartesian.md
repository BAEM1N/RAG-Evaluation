# Stage 6: Full Cartesian PreR × R × PostR (Task #37)

> **데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
>
> **8 PreR × 6 R × 8 PostR = 384 configs**
>
> **호출**: 576,000 GPT-5.4 (115,200 generation + 460,800 4-metric judge)
>
> **시간**: Step 1 retrieval matrix 10분 + Step 2 reranker (M5+AMD 병렬) 5h + Step 3 gen+judge 2.5h ≈ **7.5h 총 wallclock**
>
> **비용 (실측)**: GPT-5.4 — 추정 ~$300-400 (gen 38/s + judge 80/s)

기존 Stage 5 axis-wise(시나리오 F) 28 configs를 확장해 모든 조합(8×6×8 = 384) 평가. **상호작용 효과 탐색**.

## 1. Cartesian 매트릭스

| 축 | 항목 | 개수 |
|---|---|---:|
| **PreR** | baseline, hyde, hyde_rrf, query2doc, multi_query_para, decompose, query_expansion, query_rewrite | **8** |
| **R** | dense, bm25_kiwi, hybrid_7_3, hybrid_5_5, hybrid_3_7, hybrid_ws_5_5 | **6** |
| **PostR** | no_rerank, bge-reranker-v2-m3, bge-reranker-v2-m3-ko, ko-reranker, bge-reranker-large, qwen3-reranker-0.6b, mxbai-rerank-base-v2, **jina-reranker-m0** | **8** |
| **총** | | **384** |

### Pruning rationale (770 → 384)
- PreR: step_back, multi_query_angle 제외 (axis-wise 최하위)
- R: bm25_whitespace 제외 (gen 3.54, 명백히 망가짐)
- PostR: mxbai-large, modernReranker, bge-gemma, pixie, qwen3-4b 제외 + sigridjineth/bge-v2-minicpm/bge-v2.5-gemma2-lw/jina-v3/jina-v2-multi 호환성 실패

### Device routing (실측 timing 기반)

| Reranker | Device | 근거 |
|---|---|---|
| bge-reranker-v2-m3, bge-reranker-v2-m3-ko, ko-reranker, bge-reranker-large | **M5 Max MPS** | XLM-Roberta 계열 3× 빠름 |
| qwen3-reranker-0.6b, mxbai-rerank-base-v2, jina-reranker-m0 | **HP Z2 Mini** (AI 395+) | 다른 계열 1.25-1.5× 빠름 |
| no_rerank | (none) | top-5 = top-20[:5] truncate |

## 2. TOP 10 winners (judge mean 기준)

| 순위 | PreR | R | PostR | MRR | Hit@1 | judge_mean | sim | corr | comp | faith |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 | **query2doc** | **hybrid_7_3** | **jina-reranker-m0** | 0.7630 | 71.3% | **4.067** | 4.18 | 4.07 | 4.14 | 3.88 |
| 🥈 | query_expansion | hybrid_5_5 | jina-reranker-m0 | 0.7806 | 74.0% | 4.062 | 4.16 | 4.06 | 4.13 | 3.89 |
| 🥉 | query2doc | hybrid_5_5 | jina-reranker-m0 | 0.7726 | 73.0% | 4.036 | 4.15 | 4.03 | 4.09 | 3.87 |
| 4 | query_expansion | dense | jina-reranker-m0 | 0.7607 | 70.7% | 4.032 | 4.15 | 4.04 | 4.08 | 3.86 |
| 5 | query2doc | hybrid_5_5 | ko-reranker | 0.7401 | 68.3% | 4.030 | 4.13 | 4.04 | 4.09 | 3.86 |
| 6 | hyde | hybrid_5_5 | jina-reranker-m0 | 0.7719 | 73.0% | 4.027 | 4.16 | 4.04 | 4.06 | 3.85 |
| 7 | hyde | hybrid_7_3 | jina-reranker-m0 | 0.7756 | 73.0% | 4.026 | 4.13 | 4.02 | 4.08 | 3.87 |
| 8 | query2doc | dense | jina-reranker-m0 | 0.7558 | 70.0% | 4.026 | 4.14 | 4.02 | 4.11 | 3.83 |
| 9 | hyde_rrf | hybrid_5_5 | jina-reranker-m0 | 0.7769 | 74.0% | 4.025 | 4.12 | 4.02 | 4.06 | 3.89 |
| 10 | query_expansion | hybrid_3_7 | jina-reranker-m0 | 0.7779 | 74.3% | 4.021 | 4.13 | 4.03 | 4.07 | 3.85 |

## 2-b. TOP 5 by MRR (다른 ranking)

| 순위 | PreR | R | PostR | MRR | Hit@1 | judge | acc |
|---:|---|---|---|---:|---:|---:|---:|
| 🥇 | **multi_query_para** | **hybrid_5_5** | **jina-reranker-m0** | **0.7874** | **75.0%** | 3.991 | 0.790 |
| 🥈 | multi_query_para | hybrid_ws_5_5 | bge-reranker-v2-m3-ko | 0.7850 | 75.0% | 4.019 | 0.787 |
| 🥉 | decompose | hybrid_5_5 | bge-reranker-v2-m3-ko | 0.7816 | 74.7% | 3.967 | 0.787 |
| 4 | query_rewrite | hybrid_5_5 | bge-reranker-v2-m3-ko | 0.7812 | 74.7% | 3.971 | 0.793 |
| 5 | multi_query_para | hybrid_5_5 | bge-reranker-v2-m3-ko | 0.7809 | 74.7% | 3.974 | 0.787 |

## 2-c. TOP 10 by Accuracy (Phase 5 동일 규칙: 4-metric majority O)

| 순위 | PreR | R | PostR | MRR | judge | **acc** |
|---:|---|---|---|---:|---:|---:|
| 🥇 | **query2doc** | **hybrid_7_3** | **jina-reranker-m0** | 0.7630 | **4.067** | **0.827** |
| 🥈 | hyde_rrf | hybrid_5_5 | qwen3-reranker-0.6b | 0.7237 | 3.997 | 0.820 |
| 🥉 | baseline | hybrid_7_3 | jina-reranker-m0 | 0.7765 | 4.004 | 0.817 |
| 4 | query2doc | hybrid_7_3 | bge-reranker-large | 0.6939 | 3.991 | 0.813 |
| 5 | query2doc | hybrid_5_5 | jina-reranker-m0 | 0.7726 | 4.036 | 0.813 |
| 6 | query_expansion | hybrid_3_7 | bge-reranker-v2-m3-ko | 0.7783 | 4.005 | 0.813 |
| 7 | query_expansion | hybrid_3_7 | qwen3-reranker-0.6b | 0.7314 | 4.016 | 0.813 |
| 8 | baseline | hybrid_7_3 | bge-reranker-v2-m3 | 0.7594 | 3.988 | 0.810 |
| 9 | hyde | hybrid_5_5 | jina-reranker-m0 | 0.7719 | 4.027 | 0.810 |
| 10 | hyde_rrf | hybrid_7_3 | jina-reranker-m0 | 0.7768 | 4.007 | 0.810 |

> **Accuracy 1위 = Judge 1위** (query2doc + hybrid_7_3 + jina-reranker-m0). MRR 1위는 acc 4위 밖 — retrieval 1위와 답변 품질 1위가 다른 metric 공간임을 정량 확인.

## 2-d. Phase 5 비교 — Pipeline 최적화의 가치

본 cartesian은 Phase 5와 동일한 dataset / embedding(gemma-300m) / 4-metric majority-O 규칙 사용 → 직접 비교 가능.

| Pipeline | Generator | Accuracy |
|---|---|---:|
| 🥇 **Cartesian winner**: query2doc + Hybrid 7:3 + jina-m0 | **GPT-5.4** | **0.827** |
| Phase 5 단순 retrieval + 동일 GPT-5.4 | GPT-5.4 | 0.787 |
| Phase 5 단순 + GPT-5.4-pro (10× 비싸짐) | GPT-5.4-pro | 0.767 |
| Phase 5 Open 1위 (gpt-oss_120b / kimi-k2.5) | Open Weights | 0.740 |

→ **동일 GPT-5.4 + RAG 파이프라인 최적화만으로 +4.0pp accuracy**. GPT-5.4-pro로 모델 업그레이드한 것보다도 +6.0pp 높음. 즉 모델 더 키우는 것보다 **PreR + Hybrid + Reranker 최적 조합이 더 효과적**.

## 3. 핵심 발견 (axis-wise 대비)

### (1) **jina-reranker-m0가 cartesian winner**

| Stage | 1위 PostR | judge |
|---|---|---:|
| Stage 4-2 (retrieval만) | dragonkue/bge-v2-m3-ko (0.7697 MRR) | — |
| Stage 5 axis-wise (gen+judge) | Dongjin-kr/ko-reranker | 4.005 |
| **Stage 6 cartesian** | **jinaai/jina-reranker-m0** | **4.067** |

- jina-reranker-m0은 axis-wise에서 미실험이었음 (Stage 4-2 reranker 11개 비교 중 1개)
- 2.4B 멀티모달 reranker (Qwen2-VL 백본). 텍스트도 잘 처리
- **Top 10 중 8개가 jina-m0 사용** → 압도적 우세

### (2) query2doc의 부활 — **상호작용 효과 발견**

| PreR | axis-wise judge | cartesian top judge (with jina-m0) |
|---|---:|---:|
| query2doc | 3.967 (4위) | **4.067** (1위) |
| query_expansion | 3.998 (1위) | 4.062 |
| multi_query_para | 3.953 (8위) | 4.019 |

- axis-wise에선 query2doc가 평범했으나 jina-m0와 조합 시 **1위 등극**
- 가상문서를 잘 활용하는 reranker가 따로 있음을 시사
- **상호작용 효과는 axis-wise 단변량으론 발견 불가** — cartesian의 가치

### (3) 누적 개선 (단순 dense baseline 대비)

| Pipeline | MRR | judge | vs baseline |
|---|---:|---:|---:|
| Dense 단독 (Stage 4) | 0.6816 | (gen 미측정) | — |
| Stage 5 axis-wise winner (PreR=baseline, R=Hybrid_5_5, PostR=ko-reranker) | 0.7747 | 3.983 | retrieval +13.7% |
| **Stage 6 cartesian winner** (query2doc + Hybrid_7_3 + jina-reranker-m0) | 0.7630 | **4.067** | judge +1.5%p vs axis-wise |

→ **단변량 → cartesian 추가 개선 ~1.5%**. Marginal but meaningful.

### (4) **Bottom 10은 모두 no_rerank** — reranker 필수

| 순위 (하위) | PreR | R | PostR | judge |
|---:|---|---|---|---:|
| 384 | hyde | hybrid_ws_5_5 | no_rerank | 3.618 |
| 383 | hyde_rrf | hybrid_ws_5_5 | no_rerank | 3.623 |
| 382 | decompose | bm25_kiwi | no_rerank | 3.635 |

→ **reranker 사용 여부가 가장 큰 영향**. (위/아래 격차 0.45 = 11%)

### (5) PreR/R/PostR 영향력 정량화

판단 기준: 각 axis 변수 변화 시 judge_mean 표준편차.

| Axis | 변동 범위 (best - worst, 동일 다른축 평균) | 영향력 |
|---|---:|---|
| **PostR** | 약 0.4 (3.62 → 4.07) | **가장 큼** (10% 변동) |
| R | 약 0.15 | 중간 |
| PreR | 약 0.05 | 적음 (1%) |

→ Stage 5 결론 재확인: **PostR > R > PreR**.

## 4. 최종 winner pipeline

```
PyMuPDFLoader
  → RecursiveCharacterTextSplitter(300, 50)
  → google/embeddinggemma-300m
  → query2doc (PreR, GPT-5.4 가상문서 생성)
  → Hybrid 7:3 (FAISS dense + BM25-KIWI, RRF k=60)
  → top-20
  → jinaai/jina-reranker-m0 (멀티모달 cross-encoder, 2.4B)
  → top-5
  → GPT-5.4 RAG 답변 생성
```

**최종 metrics**: MRR 0.7630 / Hit@1 71.3% / Judge 4.067/5 / Faithfulness 3.88/5

## 5. 비용·시간

| 단계 | Time | Cost |
|---|---:|---:|
| Step 1: Retrieval matrix (48 unique top-20) | 10분 | $0 |
| Step 2a: M5 Max 4 rerankers × 48 | 4h | $0 |
| Step 2b: HP Z2 Mini 3 rerankers × 48 | 5h (병렬) | $0 |
| Step 3a: Generation 115,200 × GPT-5.4 (38/s) | ~50분 | ~$170 |
| Step 3b: Judge 460,800 × GPT-5.4 (80/s, 단일토큰) | ~100분 | ~$120 |
| **Total** | **~7.5h** | **~$290** |

(예상 $300 부합)

## 6. 레퍼런스

추가된 모델 (axis-wise 대비):
- `jinaai/jina-reranker-m0` — 멀티모달 reranker (텍스트+이미지), 2.4B, Qwen2-VL 베이스 → [HF](https://huggingface.co/jinaai/jina-reranker-m0)
- `Qwen/Qwen3-Reranker-0.6B` → [HF](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- `BAAI/bge-reranker-large` (legacy XLM-R-large) → [HF](https://huggingface.co/BAAI/bge-reranker-large)
- `mixedbread-ai/mxbai-rerank-base-v2` → [HF](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2)

### 디바이스 설정
- M5 Max: PyTorch MPS, sentence-transformers
- HP Z2 Mini (AI 395+): PyTorch 2.12 ROCm 7.13 (TheRock builds) + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`

### OpenAI 설정
- Deployment: `gpt-5.4` (gpt-5.4 v2026-03-05, GlobalStandard, 10M TPM / 100K RPM)
- Endpoint: OpenAI-compatible endpoint
- LangChain `ChatOpenAI(base_url=...)`, 100-worker ThreadPoolExecutor

## 7. 다음 단계 (선택)

- **Domain-별 분석**: 5 도메인(finance/public/medical/law/commerce) cartesian winner 분리
- **Context-type 분석**: paragraph/image/table/text 별 reranker 강점 차이
- **Stage 4-1 PreR 확장**: query2doc 변종 (gen prompt 변경, multi-doc, etc.)
- **Stage 4-2 PostR 확장**: jina-m0 변종 (jina-reranker-v2-multilingual fix 등)
- **EmbedGemma 1B**: gemma-embed-1b 새 모델 비교
