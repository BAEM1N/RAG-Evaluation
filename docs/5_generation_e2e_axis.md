# Stage 5-ext: End-to-End Generation + Judge — Axis-wise (시나리오 F)

> **데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
>
> **생성·평가 모델**: **GPT-5.4** (`gpt-5.4` deployment, OpenAI-compatible endpoint)
> via `langchain_openai.ChatOpenAI(base_url=...)`, temperature=0
>
> **고정 winner pipeline**:
> - Loader: `PyMuPDFLoader`
> - Parser: `RecursiveCharacterTextSplitter(300, 50)`
> - Embedding: `google/embeddinggemma-300m`
> - Retriever (default): Hybrid 3:7 (FAISS + BM25-KIWI)
> - Reranker (default): `dragonkue/bge-reranker-v2-m3-ko` (top-20 → top-5)
>
> **Judge**: 4-metric (similarity / correctness / completeness / faithfulness), 1–5 정수, single-token output (Phase 5 동일 rubric, `scripts/llm_judge.py::EVAL_PROMPTS`)
>
> **호출**: 28 configs × 300q × (1 gen + 4 judge) = 42,000 GPT-5.4 호출, 30-worker `ThreadPoolExecutor`, 약 2시간

각 axis에서 한 component만 변수, 나머지는 winner 고정. **검색 metric vs 생성 품질**의 괴리를 정량화.

> 본 문서는 기존 `5_generation.md` (Phase 5, Open vs Closed weights 분석) 의 후속.
> Stage 1-4-2 단변량 실험 winner들을 고정한 채 **생성·평가까지 end-to-end** 측정한 별도 실험.

## 1. Axis A — Pre-retriever (10 strategies)

> 고정: R = Hybrid 3:7, PostR = `bge-reranker-v2-m3-ko`

| 순위 | PreR | MRR | Hit@1 | judge | sim | corr | comp | faith |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 | **query_expansion** | 0.7783 | 70.0% | **3.998** | 4.10 | 4.00 | 4.03 | 3.85 |
| 2 | hyde | 0.7738 | 70.3% | 3.985 | 4.10 | 4.00 | 4.01 | 3.84 |
| 3 | decompose | 0.7786 | 70.7% | 3.967 | 4.08 | 3.98 | 3.98 | 3.83 |
| 4 | **query2doc** | 0.7768 | 70.0% | 3.967 | 4.07 | 3.98 | 4.00 | 3.81 |
| 5 | hyde_rrf | 0.7713 | 70.3% | 3.965 | 4.09 | 3.98 | 3.98 | 3.82 |
| 6 | multi_query_angle | 0.7770 | 69.3% | 3.962 | 4.08 | 3.97 | 4.00 | 3.80 |
| 7 | query_rewrite | 0.7741 | 70.3% | 3.962 | 4.06 | 3.96 | 3.98 | 3.84 |
| 8 | multi_query_para | 0.7780 | 70.3% | 3.953 | 4.06 | 3.96 | 3.99 | 3.81 |
| 9 | **baseline** | 0.7697 | 70.7% | 3.953 | 4.05 | 3.97 | 3.96 | 3.81 |
| 10 | step_back | 0.7692 | 69.3% | 3.941 | 4.04 | 3.96 | 3.97 | 3.78 |

**결론**: 모든 PreR 전략이 judge 3.94~4.00 (편차 1.5%). **Pre-retriever는 생성 품질에 거의 영향 없음**.
- 단, **query_expansion** (키워드 추가)이 검색 metric도 좋고 (MRR 0.7783, +1.1pp vs baseline) generation도 1위 — **소소한 효과 존재**.
- step_back (검색 metric 최하)이 생성 metric에서도 최하 → **검색-생성 일치**.

> 흥미로운 반전: Stage 4-1 단변량(검색 metric만)에선 거의 모든 PreR이 baseline에 패배했으나, **post-rerank 후 생성까지 보면 query_expansion / hyde / decompose 등이 baseline을 미세하게 능가**. Reranker가 PreR로 인한 retrieval noise를 보정해주는 역할.

## 2. Axis B — Retriever (7 strategies)

> 고정: PreR = baseline, PostR = `bge-reranker-v2-m3-ko`

| 순위 | R | MRR | Hit@1 | judge | sim | corr | comp | faith |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 | **Hybrid 5:5** | 0.7747 | 70.7% | **3.983** | 4.10 | 3.99 | 4.01 | 3.83 |
| 2 | Dense | 0.7793 | 70.0% | 3.953 | 4.04 | 3.96 | 3.99 | 3.81 |
| 3 | Hybrid 7:3 | 0.7783 | 70.3% | 3.938 | 4.05 | 3.94 | 3.97 | 3.80 |
| 4 | BM25 + KIWI | 0.7680 | 70.0% | 3.923 | 4.03 | 3.94 | 3.94 | 3.78 |
| 5 | **Hybrid 3:7** | 0.7697 | 70.3% | 3.922 | 4.05 | 3.92 | 3.94 | 3.77 |
| 6 | Hybrid-WS 5:5 | 0.7724 | 70.0% | 3.922 | 4.02 | 3.93 | 3.93 | 3.80 |
| 7 | BM25 whitespace | 0.6750 | 56.0% | **3.540** | 3.64 | 3.53 | 3.53 | 3.46 |

**결론**:
- **Hybrid 5:5가 생성 품질 1위 (3.98)** — 검색 단변량에서 Hybrid 3:7이 winner였지만, 생성까지 보면 균형이 좋음.
- BM25 whitespace는 검색 metric도 (0.675) generation도 (3.54) 최악 — **KIWI 형태소 분석 필수성 재확인**.
- Dense / Hybrid 7:3 / Hybrid 3:7 / Hybrid 5:5 모두 생성 quality 차이 0.06 이내 — **상위 retriever 간 generation 거의 동률**.

## 3. Axis C — Post-retriever / Reranker (11 strategies)

> 고정: PreR = baseline, R = Hybrid 3:7

| 순위 | Reranker | MRR | Hit@1 | judge | sim | corr | comp | faith |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 | **`Dongjin-kr/ko-reranker`** | 0.7320 | 67.3% | **4.005** | 4.13 | 4.01 | 4.03 | 3.85 |
| 2 | **`dragonkue/bge-reranker-v2-m3-ko`** | 0.7697 | 74.0% | 3.978 | 4.09 | 3.97 | 3.99 | 3.86 |
| 3 | `Qwen/Qwen3-Reranker-4B` | 0.7508 | 70.3% | 3.970 | 4.08 | 3.99 | 3.99 | 3.82 |
| 4 | `mxbai-rerank-base-v2` | 0.7319 | 67.3% | 3.949 | 4.05 | 3.95 | 3.98 | 3.82 |
| 5 | `Qwen/Qwen3-Reranker-0.6B` | 0.7255 | 66.3% | 3.945 | 4.05 | 3.96 | 3.97 | 3.79 |
| 6 | `BAAI/bge-reranker-v2-m3` | 0.7488 | 70.0% | 3.931 | 4.03 | 3.94 | 3.98 | 3.78 |
| — | **no_rerank (baseline)** | 0.7171 | 65.3% | 3.869 | 3.95 | 3.88 | 3.88 | 3.77 |
| 7 | `mxbai-rerank-large-v2` | 0.6945 | 66.7% | 3.717 | 3.81 | 3.72 | 3.74 | 3.60 |
| 8 | `naver/modernReranker` | 0.5159 | 41.7% | 3.402 | 3.50 | 3.42 | 3.38 | 3.31 |
| 9 | `BAAI/bge-reranker-v2-gemma` | 0.4152 | 28.0% | 3.171 | 3.28 | 3.17 | 3.13 | 3.10 |
| 10 | `telepix/PIXIE-Spell-Reranker` | 0.3276 | 21.0% | 2.969 | 3.09 | 2.96 | 2.92 | 2.90 |

**결론**:
- **Generation 1위 = `Dongjin-kr/ko-reranker` (3.99) vs Retrieval 1위 = `bge-reranker-v2-m3-ko` (0.7697 MRR)** — **둘 다 한국어 fine-tune, 평가 metric에 따라 우열 갈림**.
- 단순 dense 검색(Hit@1=65%) → ko-reranker(67%) 한 단계 + judge +0.13 (+3.4%) 개선.
- Korean fine-tune 두 모델이 모든 신모델·SOTA·multilingual 모델을 압도.
- 안 좋은 reranker는 baseline보다 **악화** (mxbai-large −0.15, bge-gemma −0.70, pixie −0.90).

## 4. 핵심 통합 관찰

### (1) 생성 품질과 검색 metric은 일치, 그러나 순위는 다를 수 있음

| Axis | Retrieval 1위 | Generation 1위 | 일치? |
|---|---|---|---|
| A PreR | decompose / multi_query_para | query_expansion | 다름 (인접) |
| B R | Dense | Hybrid 5:5 | 다름 (인접) |
| C PostR | bge-v2-m3-ko | ko-reranker | 다름 (둘 다 KR) |

→ **상관관계는 강하지만 1위 결정은 generation까지 봐야 함**.

### (2) "안 좋은 컴포넌트"는 양쪽에서 치명적

- bm25_whitespace (검색 0.675 → 생성 3.54)
- pixie reranker (검색 0.328 → 생성 2.97)
- bge-gemma (검색 0.415 → 생성 3.17)

→ 검색 metric으로 "사용 금지" 판정 가능.

### (3) Pre-retriever는 생성에 영향 미미 (편차 1.5%)

→ 한국어 단답형 RAG에선 **PreR 단계 생략 또는 query_expansion 정도면 충분**.

### (4) Reranker가 생성 품질에 가장 큰 효과

- 최고 reranker (ko-reranker) vs no_rerank: judge **+0.13** (+3.5%)
- 최고 retriever (Hybrid 5:5) vs 차하위 (Hybrid 7:3): judge **+0.04** (+1.1%)
- 최고 PreR (query_expansion) vs baseline: judge **+0.05** (+1.1%)

→ **PostR > R > PreR** (생성 품질 영향 순서).

### (5) 한국어 RAG에서 winner pipeline (재확정)

```
PyMuPDFLoader
  → RecursiveCharacterTextSplitter(300, 50)
  → google/embeddinggemma-300m
  → Hybrid 5:5 (dense + BM25-KIWI, RRF k=60)   ← Hybrid 3:7 에서 5:5 로 갱신
  → top-20
  → Dongjin-kr/ko-reranker (top-5)             ← bge-v2-m3-ko 에서 ko-reranker 로 갱신
  → GPT-5.4 RAG 답변
```

**기존 Stage 4/4-2 winner**(Hybrid 3:7 + bge-v2-m3-ko)는 **검색 metric 최적**.
**Stage 5 e2e winner**(Hybrid 5:5 + ko-reranker)는 **생성 품질 최적**.
→ 운영 시 두 옵션 모두 사용 가능, 차이는 ~3% 수준.

## 5. 비용

| 항목 | 호출 수 | 토큰 (대략) | GPT-5.4 비용 |
|---|---:|---:|---:|
| 생성 (28 × 300) | 8,400 | ~10M | ~$13 |
| Judge (28 × 300 × 4) | 33,600 | ~14M | ~$8 |
| **합계** | 42,000 | ~24M | **~$21** |

> 시나리오 F 예상 $29 대비 27% 절감 (cache hit + 짧은 응답 + 30-worker batching).

## 6. 레퍼런스

### 모델
- GPT-5.4 (gpt-5.4) → [docs](https://platform.openai.com/docs/)
- LangChain `ChatOpenAI` (OpenAI-compatible API 호환 모드) → [API ref](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)

### 평가 방법
- 4-metric LLM-as-Judge rubric — Phase 5 동일 (`scripts/llm_judge.py::EVAL_PROMPTS`)
- 1-5 정수 점수, single-token output (Liu et al. 2023 G-Eval [arXiv:2303.16634](https://arxiv.org/abs/2303.16634) 류 방식)

### 기타 stage 문서
- [1_loader.md](./1_loader.md) — Loader (pymupdf)
- [2_parser.md](./2_parser.md) — Parser (LC Recursive 300/50, semantic chunker 모두 패배)
- [3_embedding.md](./3_embedding.md) — Embedding (KoE5 / gemma-300m)
- [4_retriever.md](./4_retriever.md) — Retriever (Hybrid 3:7 KIWI)
- [4-1_pre_retriever.md](./4-1_pre_retriever.md) — Pre-retriever (10 strategies, baseline 강세)
- [4-2_post_retriever.md](./4-2_post_retriever.md) — Post-retriever (11 reranker, dragonkue/bge-v2-m3-ko)
- [5_generation.md](./5_generation.md) — Phase 5 (Open vs Closed weights, 46 generation 모델)
- [EXPERIMENT_PLAN.md](./EXPERIMENT_PLAN.md) — 종합 실험 계획서

## 7. 다음 단계

- ✅ Axis-wise 생성+평가 완료
- ⏳ Full cartesian (10 PreR × 7 R × 11 PostR = 770 configs, 약 $400-600) — 상호작용 효과 측정용
- ⏳ Domain-별 분석 (finance/public/medical/law/commerce 5 도메인 break-down)
- ⏳ Context-type별 분석 (paragraph/image/table/text)
