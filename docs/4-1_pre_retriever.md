# Pre-retriever — 검색 이전 query 변형

> **데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
>
> **고정 파이프라인** (Stage 1~4 winner):
> - Loader: `PyMuPDFLoader`
> - Parser: `RecursiveCharacterTextSplitter(300, 50)`
> - Embedding: `google/embeddinggemma-300m` (HuggingFaceEmbeddings)
> - Retriever: `EnsembleRetriever`(FAISS dense + BM25-KIWI, weights=[0.3, 0.7]) — Hybrid RRF k=60
> - Top-k = 5
>
> **LLM (query 변형용)**: **GPT-5.4** (`gpt-5.4` deployment, OpenAI-compatible endpoint), via `langchain_openai.ChatOpenAI(base_url=...)`
>
> **호출**: 2,400 unique LLM calls, 20-worker `ThreadPoolExecutor` 병렬화, 약 5분, ~$3.15

검색 엔진에 query를 보내기 전 query를 가공하거나 추가 query를 생성해 recall을 끌어올리는 단계.

> **⚠️ 이 문서의 순위 = retrieval-only(MRR) 단변량 측정** (reranker·생성 없음, Stage 1~4 고정). 이 기준 winner 는 `multi_query_para`(MRR 0.7189)지만, 전략 간 차이는 noise(±0.002) 범위다.
> **파이프라인에 실제 채택된 Stage 4-1 winner 는 `query_expansion`** — reranker 적용 후 end-to-end **judge 점수** 1위(3.998)이기 때문이다. 검색지표가 아닌 생성 품질로 결정한 근거는 [`5_generation_e2e_axis.md`](./5_generation_e2e_axis.md) 및 [`REPORT.md`](./REPORT.md) §6 참조.

## 1. 10 전략 비교 결과 (retrieval-only MRR)

| 순위 | 전략 | MRR | Hit@1 | Hit@5 | File@5 | vs baseline (MRR) |
|---:|---|---:|---:|---:|---:|---:|
| 🥇 | **multi_query_para** | **0.7189** | **66.0%** | 81.0% | 91.7% | **+0.0018** |
| — | baseline (no PreR) | 0.7171 | 65.3% | 80.3% | 91.7% | (기준) |
| 3 | hyde_rrf | 0.7159 | 64.7% | 80.7% | 91.7% | −0.0012 |
| 4 | hyde | 0.7124 | 64.3% | 81.0% | **92.0%** | −0.0047 |
| 5 | decompose | 0.7111 | 63.7% | **81.3%** | 91.7% | −0.0060 |
| 6 | query_expansion | 0.7076 | 63.3% | 81.0% | **92.0%** | −0.0095 |
| 7 | query2doc | 0.6988 | 63.0% | 80.3% | 91.3% | −0.0183 |
| 8 | query_rewrite | 0.6951 | 61.7% | 80.3% | 91.7% | −0.0220 |
| 9 | multi_query_angle | 0.6434 | 54.7% | 78.7% | 91.7% | **−0.0737** |
| 10 | step_back | 0.6032 | 50.3% | 75.7% | 91.3% | **−0.1139** |

## 2. 핵심 관찰 (네거티브 결과 중심)

### (1) 거의 모든 pre-retriever가 baseline에 패배

10개 전략 중 baseline을 이긴 건 **multi_query_para 하나뿐**, 그것도 MRR +0.0018 (Hit@1 +0.7pp) — **실질 의미 없는 noise 범위**. 8개는 오히려 성능 저하.

→ 이 데이터셋 (단답형 factoid QA, query가 이미 구체적)에서는 **원 query가 이미 거의 최적**.

### (2) Step-back / Multi-query-angle 의 추상화는 치명적

| 전략 | MRR 감소 | 원인 추정 |
|---|---:|---|
| step_back | −0.114 | 추상화로 정답 키워드 제거, query specificity 손실 |
| multi_query_angle | −0.074 | "추상" / "다른 어휘" 변형이 점수 평균을 끌어내림 |

추상 query는 retrieval 단계에서 **노이즈 chunk가 함께 RRF에 합쳐져 정답의 상대 순위를 떨어뜨림**.

### (3) HyDE / Query2doc — 약한 음의 효과

| 전략 | MRR | 비고 |
|---|---:|---|
| hyde | 0.7124 | dense=가상답안, BM25=원 query — 정답 키워드는 보존되지만 dense 임베딩이 환각 답안 쪽으로 끌림 |
| hyde_rrf | 0.7159 | 원 query + 가상답안 둘 다 RRF → 손해 최소화 (−0.001) |
| query2doc | 0.6988 | 가상문서 concat이 BM25 쪽에도 노이즈 토큰 추가 → 손해 큼 |

**HyDE 패밀리는 원 query를 살려두는 RRF 변형(hyde_rrf)이 가장 안전**.

### (4) Query rewriting / expansion — 미미한 negative

| 전략 | MRR | 비고 |
|---|---:|---|
| query_expansion | 0.7076 | 키워드 5개 추가 → BM25는 약간 도움, dense 임베딩에는 노이즈 |
| query_rewrite | 0.6951 | 단일 명확화 쿼리로 대체 → 원 query의 정확한 키워드가 빠지면서 약화 |

### (5) Decompose — 추가 sub-question은 의미 없음

decompose (0.7111) 는 baseline 보다 약간 낮음. allganize 단답 dataset에서는 multi-hop이 거의 없어 sub-question 분해가 효과 없음. 멀티홉 dataset에서는 다를 수 있음.

### (6) File@5는 거의 모든 전략에서 91-92% — 정답 file 자체는 잘 찾음

차이는 **page-level 정확도**에만 나타남. retrieval pool은 충분히 좋고, 문제는 정확한 chunk 1위 결정.

## 3. 결론

> **이 데이터셋에서는 pre-retriever를 쓰지 않는 게 합리적**.
> 굳이 쓴다면 `multi_query_para` (noise 범위 내 +0.0018, LLM 비용 발생) 또는 `hyde_rrf` (안전한 RRF 백오프).

### 다음 stage(생성+평가)에서 PreR 변수 후보 (cartesian top-3)

검색 metric 기준 상위 3개:
1. **baseline** (LLM 0회, 비용 0) — 가장 강력
2. **multi_query_para** (+0.0018 MRR, 300 LLM call/실험)
3. **hyde_rrf** (−0.0012 MRR, 300 LLM call/실험)

생성·평가에서는 retrieval MRR과 다른 결과 나올 수 있음 (예: HyDE가 답안 생성에는 더 풍부한 컨텍스트 제공). axis-wise 생성 평가 (시나리오 A, $10)로 검증 예정.

## 4. 향후 검토 (현재 미실험)

| 전략 | 메모 |
|---|---|
| **GenPRF** (Generative Pseudo-Relevance Feedback) | LLM이 1차 retrieval 결과를 보고 query 확장 (multi-stage) |
| **CoT-based retrieval planning** | LLM이 검색 계획을 step-by-step 출력, 단계마다 검색 |
| **Iterative retrieval** | 검색 → 부족하면 다시 query 변형 (Self-Ask 스타일 반복) |
| **Conversational rewriting** | 대화 문맥이 있는 경우 (현 dataset은 단발성 질문) |
| **Domain-specific prompts** | finance/medical/law 도메인 분류 후 도메인별 HyDE 프롬프트 |

## 5. 레퍼런스

### 원논문
- **HyDE** (Hypothetical Document Embeddings) — Gao, Ma, Lin, Callan. "Precise Zero-Shot Dense Retrieval without Relevance Labels" (ACL 2023) → [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) · [ACL Anthology](https://aclanthology.org/2023.acl-long.99/)
- **Query2doc** — Wang, Yang, Wei. "Query2doc: Query Expansion with Large Language Models" (EMNLP 2023) → [arXiv:2303.07678](https://arxiv.org/abs/2303.07678) · [ACL Anthology](https://aclanthology.org/2023.emnlp-main.585/)
- **Step-back Prompting** — Zheng, Mishra, Chen, et al. (Google DeepMind). "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models" (ICLR 2024) → [arXiv:2310.06117](https://arxiv.org/abs/2310.06117)
- **Self-Ask (Decomposition)** — Press, Zhang, Min, Schmidt, Smith, Lewis. "Measuring and Narrowing the Compositionality Gap in Language Models" (EMNLP 2023 Findings) → [arXiv:2210.03350](https://arxiv.org/abs/2210.03350)
- **Query Expansion with LLM** — Jagerman, Zhuang, Qin, Wang, Bendersky (Google). "Query Expansion by Prompting Large Language Models" (2023) → [arXiv:2305.03653](https://arxiv.org/abs/2305.03653)
- **Query Rewriting** — Ma, Gong, He, Zhao, Duan (Microsoft). "Query Rewriting for Retrieval-Augmented Large Language Models" (EMNLP 2023) → [arXiv:2305.14283](https://arxiv.org/abs/2305.14283) · [ACL Anthology](https://aclanthology.org/2023.emnlp-main.322/)

### 라이브러리
- **LangChain MultiQueryRetriever** → [docs](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) · [API ref](https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever)
- **LangChain ChatOpenAI** (OpenAI-compatible API 호환) → [docs](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- **Kiwi 형태소 분석기** → [github.com/bab2min/Kiwi](https://github.com/bab2min/Kiwi)
- **Reciprocal Rank Fusion (RRF)** — Cormack et al. SIGIR 2009 → [DOI](https://doi.org/10.1145/1571941.1572114)

### 모델
- **GPT-5.4** (gpt-5.4) → [OpenAI docs](https://platform.openai.com/docs/)
- **EmbeddingGemma 300m** → [HuggingFace](https://huggingface.co/google/embeddinggemma-300m) · [Google AI docs](https://ai.google.dev/gemma/docs/embeddinggemma)

## 6. 다음 단계

- ✅ Stage 4-1 검색 metric 단변량 — 완료 (multi_query_para winner, baseline 동급)
- ⏳ Stage 4-2 확장 — 신 reranker 9종 + non-reranker 전략 4개 (LangChain wrapper로 통일)
- ⏳ Cartesian (PreR × R × PostR) — 시나리오 D 또는 E (~$30~$820)
- ⏳ axis-wise 생성+평가 (시나리오 A, $10) — retrieval MRR vs generation quality 괴리 측정
