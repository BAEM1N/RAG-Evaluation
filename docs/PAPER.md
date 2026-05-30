# 한국어 RAG 파이프라인의 컴포넌트별 효과 분해 — 단변량과 Cartesian 분석을 통한 실증 연구

**저자**: BAEM1N
**데이터셋**: [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
**Repository**: <https://github.com/BAEM1N/RAG-Evaluation>
**HuggingFace**: [BAEM1N/Korean-RAG-LLM-Judge-Benchmark](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark)
**기간**: 2026-04 ~ 2026-05

---

## 초록

본 연구는 한국어 Retrieval-Augmented Generation (RAG) 파이프라인의 각 컴포넌트가 최종 답변 품질에 미치는 효과를 정량적으로 분해한다. allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs × 5 도메인)를 사용해 (1) 단변량 비교(6 stage, 총 95+ 구성 요소), (2) end-to-end axis-wise 평가(28 configs), (3) Cartesian 전수 탐색(384 configs = 8 PreR × 6 R × 8 PostR)을 수행하여 약 1.2M LLM 호출의 평가 데이터를 산출했다. 주요 발견은 다음과 같다. (i) 단순한 character-level recursive splitter(300/50)가 LLM-기반 chunker 등 39종 비교에서 winner임을 확인했다. (ii) 한국어 fine-tune reranker(dragonkue/bge-reranker-v2-m3-ko)가 2025 SOTA multilingual reranker(Qwen3-Reranker-4B, 약 6.7배 큰 모델)를 +1.83pp MRR 능가했다. (iii) Pre-retriever 단변량 효과는 미미(±1%)했으나 Cartesian에서 query2doc × jina-reranker-m0 상호작용을 통해 답변 품질 1위(judge 4.067)에 도달했다. (iv) 컴포넌트 영향력 순서는 PostR > R > PreR이며, 가장 단순한 baseline 대비 누적 개선은 MRR +15.5%, Hit@1 +16.0pp(+27.1% 상대), judge +5.6%였다.

**키워드**: Korean RAG, retrieval evaluation, reranker, LLM-as-judge, cartesian benchmark

---

## 1. 서론

### 1.1 배경

Retrieval-Augmented Generation은 대형 언어 모델의 hallucination과 outdated knowledge 문제를 완화하는 표준 패러다임이 됐다 [Lewis 2020]. RAG 시스템은 일반적으로 (a) Document Loader, (b) Text Splitter (Parser), (c) Embedding Model, (d) Vector Retriever, (e) (선택적) Pre-retriever 및 Post-retriever, (f) Generator LLM의 6개 컴포넌트로 구성된다. 각 컴포넌트는 독립적으로 발전했고 대안이 폭넓게 존재한다.

영어권에서는 RAG 컴포넌트 비교 벤치마크가 다수 보고됐으나 [LangChain Eval, RAGAS 등], **한국어 RAG**에 대한 통합적·정량적 분석은 부족하다. 특히 (i) 모든 컴포넌트가 동시 변동 가능한 cartesian 비교, (ii) 검색 metric(MRR/Hit@k)과 생성 품질(LLM-as-Judge) 양쪽을 모두 측정한 사례는 드물다.

### 1.2 연구 질문

본 연구는 다음 4가지를 검증한다.

- **RQ1**: 한국어 RAG 파이프라인에서 각 컴포넌트가 retrieval 성능(MRR)과 generation 품질(judge)에 미치는 단변량 효과는?
- **RQ2**: 컴포넌트 간 상호작용 효과는 존재하며, cartesian 평가는 단변량 평가 대비 얼마나 추가 가치를 제공하는가?
- **RQ3**: 한국어 특화 모델 vs 일반 multilingual 모델 / 한국어 특화 vs 일반 도메인 fine-tune의 trade-off?
- **RQ4**: 가장 단순한 dense baseline 대비 누적 최적화가 가능한 한계는?

### 1.3 기여

1. 한국어 RAG의 컴포넌트별 효과를 단변량 + axis-wise + 전 Cartesian 3 단계로 분해한 첫 통합 벤치마크
2. 95+ 구성 요소 비교 (7 loader / 42 chunker / 27 embedding / 7 retriever / 10 PreR / 25 reranker / 46 generator) 및 580K+ LLM-as-Judge 평가
3. 한국어 fine-tune reranker가 2025 SOTA multilingual reranker를 능가함을 실증
4. Pre-retriever와 reranker 간 상호작용(query2doc × jina-reranker-m0)을 발견, Cartesian 평가의 필요성 입증
5. 전체 실험 코드·데이터·결과를 MIT 라이선스로 공개

---

## 2. 관련 연구

### 2.1 RAG 평가 데이터셋

- **allganize/RAG-Evaluation-Dataset-KO**: 한국어 5 도메인(금융/공공/의료/법률/커머스) 300 Q&A × 58 PDFs. 본 연구의 기본 데이터셋.
- **MTEB-Korean**: 다양한 검색·분류 task 포함하지만 RAG-specific Q&A는 제한적.
- **KoBEST**: 한국어 NLU 벤치마크, RAG 미포함.

### 2.2 RAG 컴포넌트 평가

- Pre-retriever: HyDE [Gao 2022], Query2doc [Wang 2023], Step-back [Zheng 2023], Self-Ask [Press 2022], Query Rewriting [Ma 2023].
- Retriever: BM25 [Robertson 2009], Hybrid RRF [Cormack 2009].
- Reranker: BGE-M3 [Chen 2024], Qwen3-Reranker [2025], jina-reranker series.
- Generator: Closed (GPT, Claude, Gemini) vs Open Weights (Qwen, DeepSeek, Kimi, Llama).
- 평가 방법: LLM-as-Judge [Liu 2023 G-Eval], multi-judge consensus.

### 2.3 한국어 임베딩 / Reranker

- KoE5 [Jang 2024]: 한국어 retrieval 임베딩.
- Korean fine-tune reranker: dragonkue, Dongjin-kr, upskyy, naver/modernReranker.
- 본 연구는 이들을 일반 multilingual 모델과 동일 조건에서 비교.

---

## 3. 실험 설계

### 3.1 데이터셋

| 항목 | 값 |
|---|---|
| 질문 수 | 300 (도메인별 60: finance / public / medical / law / commerce) |
| PDF 수 | 58 (다양한 길이 1-50+ 페이지) |
| Context type | paragraph 148, image 57, table 50, text 45 |
| Ground truth | 정답 텍스트 + 정답 PDF 파일명 + 페이지 번호 |

### 3.2 평가 metric

| Metric | 정의 |
|---|---|
| **MRR** | $\frac{1}{N}\sum_i \frac{1}{rank_i}$ — 정답 chunk의 역순위 평균 |
| **Hit@k** | top-k에 정답 chunk가 포함된 비율 |
| **File@k** | top-k에 정답 file이 포함된 비율 (loose 매칭) |
| **judge** | LLM 4-metric 평균 (1-5 정수): similarity, correctness, completeness, faithfulness |

LLM-as-Judge는 GPT-5.4 (reasoning_effort=none) 단일 모델 4-metric 분리 평가 방식을 사용했다. 메트릭별 rubric은 `scripts/llm_judge.py::EVAL_PROMPTS` 참조.

### 3.3 컴포넌트 비교 매트릭스

| Stage | 비교 수 | 카테고리 |
|---|---:|---|
| 1. Loader | 7 | pymupdf, pdfplumber, pymupdf4llm, pdfminer, docling, pypdf, opendataloader |
| 2. Parser (Chunker) | 42 | char-based 32 (LangChain / Chonkie / LlamaIndex × size grid) + semantic / LLM-based 10 |
| 3. Embedding | 27 | KoE5, EmbeddingGemma, BGE-M3, Qwen3-Embed, Jina, Snowflake, Nomic, harrier 등 |
| 4. Retriever | 7 | Dense / BM25-KIWI / BM25-whitespace / Hybrid 7:3, 5:5, 3:7, Hybrid-WS 5:5 |
| 4-1. Pre-retriever | 10 | baseline, hyde, hyde_rrf, query2doc, multi-query (para/angle), step-back, decompose, query_expansion, query_rewrite |
| 4-2. Post-retriever (Reranker) | 25 | dragonkue, shoxa-mir, jina-m0, Qwen3-Reranker (0.6B/4B), bge-v2-m3, ko-reranker 등 |
| 5. Generator (Phase 5) | 46 | Open 27 + Closed 19 — gpt-oss, Kimi, DeepSeek, GPT-5.4, Claude Opus 4.7, Gemini-3.1-pro 등 |

### 3.4 평가 단계

본 연구는 3 단계로 진행된다.

**Stage 1-4-2**: 각 컴포넌트를 단변량으로 비교 (다른 컴포넌트는 default 고정). retrieval metric (MRR/Hit/File) 측정.

**Stage 5 axis-wise (시나리오 F)**: Stage 1-4-2 winner들을 고정한 채 각 axis만 변화 (28 configs = 10 + 7 + 11). 검색 metric에 추가로 LLM-as-Judge 4-metric 측정.

**Stage 6 Cartesian (Task #37)**: 8 PreR × 6 R × 8 PostR = 384 configs. 모든 조합 평가하여 상호작용 효과 탐색.

### 3.5 인프라

| 장비 | 용도 |
|---|---|
| MacBook Pro 14 (M5 Max, 128GB) | 임베딩 + reranker (M5 MPS) + 메인 driver |
| HP Z2 Mini G1a (AMD Ryzen AI Max+ 395, 128GB) | Reranker (AI 395+ ROCm 7.13) |
| DGX Spark (GB10, 128GB) | Phase 5 local LLM (ollama) |
| OpenAI-compatible API (gpt-5.4) | Stage 4-1/5/6 LLM 변형 + 생성 + 평가 |
| 다중 vendor API | Phase 5 Closed 19 생성 + 3-judge ensemble |

전체 파이프라인은 **LangChain LCEL**로 구현됐다 (`PyMuPDFLoader`, `RecursiveCharacterTextSplitter`, `FAISS`, `BM25Retriever`, `EnsembleRetriever`, `CrossEncoderReranker`, `ChatOpenAI(base_url=...)`). Cartesian의 reranker 추론은 M5 Max + AMD AI 395+ 분산 처리됐다 (XLM-Roberta 계열은 M5, Qwen3/jina-m0/mxbai는 AMD).

---

## 4. 결과

### 4.1 Stage 1: Loader (RQ1)

7종 PDF Loader를 동일 chunking(1000/200) + embedding(gemma-300m) + dense retrieval로 평가.

| Loader | MRR | Hit@1 | Parse(s) |
|---|---:|---:|---:|
| **pymupdf** | **0.6486** | 57.0% | **3.1** |
| pdfplumber | 0.6468 | 56.3% | 108.8 |
| pymupdf4llm | 0.6388 | 54.7% | 547.5 |
| pdfminer | 0.6301 | 54.7% | 144.9 |
| docling | 0.6241 | 54.7% | 1,162.5 |
| pypdf | 0.6203 | 53.3% | 32.9 |
| opendataloader | 0.5993 | 50.0% | 169.3 |

→ **단순 평문 추출(pymupdf)이 winner.** Markdown 변환(pymupdf4llm), OCR+layout 분석(docling)은 비용 대비 효과 없음. **MRR 1-7위 격차 약 5pp (한국어 평문 추출 정확도 평준화)**.

### 4.2 Stage 2: Parser (RQ1)

42종 chunker 비교 (2 그룹).

**그룹 A — Char-based 32종 (Dense baseline)**: Library × chunk_size 매트릭스.

| Chunker | size | MRR |
|---|---|---:|
| **Chonkie Fast** | 800 | **0.6903** |
| Chonkie Recursive | 300 | 0.6885 |
| Chonkie Sentence | 300/50 | 0.6881 |
| LC Recursive | 300/50 | 0.6816 |
| LC Token (cl100k) | 256 | 0.6798 |
| Chonkie Token (gpt2) | 256 | **0.4193** ❌ |

**chunker마다 sweet spot이 다르다**: LC/Chonkie Recursive·Sentence는 300/50, Chonkie Fast는 800, LlamaIndex Sentence는 500/100. Tokenizer 차이가 결정적: 같은 256-token chunk가 cl100k 기반에서는 0.6798이지만 gpt2 기반에서는 한국어를 byte-level로 잘게 분해해 0.4193으로 폭락한다.

**그룹 B — Semantic + LLM-based 10종 (Hybrid 3:7 baseline)**: Embedding/LLM 호출이 필요한 비싼 chunker.

| Chunker | MRR | Parse |
|---|---:|---:|
| (기준) LC Recursive 300/50 hybrid 재측정 | **0.7171** | 5s |
| Chonkie Slumber (gpt-5.4, LLM-기반) | 0.7112 | **5,608s** |
| LlamaIndex SemanticSplitter | 0.7076 | 213s |
| Chonkie Semantic (text-embedding-3-large) | 0.6948 | 2,191s |
| LC MarkdownHeaderTextSplitter | 0.6574 | 1,083s |

→ **단순한 char-based가 모든 semantic / LLM-based chunker를 능가**. Slumber는 그룹 B 1위지만 char-based winner 대비 −0.59pp + 1100× parse time. 한국어 단답형 RAG에서 의미 경계 chunking은 비용 대비 효과 없음.

### 4.3 Stage 3: Embedding (RQ3)

27 모델 leaderboard.

| 순위 | 모델 | dim | MRR | 특이사항 |
|---:|---|---:|---:|---|
| 🥇 | KoE5 | 1024 | **0.6871** | 한국어 fine-tune |
| 🥈 | gemma-embed-300m | 768 | 0.6650 | 최고 소형 |
| 🥉 | pixie-rune-v1 | 1024 | 0.6627 | |
| 4-5 | snowflake-arctic-ko / l-v2 | 1024 | 0.66 | 한국어 튜닝 |
| 17 | qwen3-embed-8b | 4096 | 0.5271 | 영어 우세 |

→ **소형 한국어 fine-tune이 대형 영어 우세 모델을 압도**: KoE5 (1024d) > Qwen3-Embed-8B (4096d), +0.16 MRR. 한국어 특화 4개 모델(koe5, snowflake-arctic-ko, kure-v1, pixie-rune-v1)이 Top 8에 모두 포진.

### 4.4 Stage 4: Retriever (RQ1)

| 전략 | MRR | Hit@1 |
|---|---:|---:|
| **Hybrid 3:7 (Dense + BM25-KIWI, RRF k=60)** | **0.7171** | 65.3% |
| Hybrid 5:5 | 0.7137 | 65.3% |
| Dense alone | 0.6816 | 59.0% |
| BM25 + KIWI alone | 0.6783 | 61.3% |
| BM25 + whitespace | 0.5344 | 48.3% |

→ **Hybrid가 단독 검색을 모두 능가** (Hybrid 3:7 dense vs Dense alone +3.55pp). **KIWI 형태소 분석 필수**: BM25-KIWI vs BM25-whitespace +14.4pp. Dense와 BM25-KIWI 단독 성능 거의 동률 — 두 방식 모두 한국어에서 효과적이며 hybrid가 최강.

### 4.5 Stage 4-1: Pre-retriever (RQ1)

Stage 4 winner + Stage 4-2 winner 고정한 채 PreR만 변경. LLM = GPT-5.4. winner 는 **end-to-end judge** 기준(MRR 은 retrieval-only 보조 지표, 전략 간 차이 ±0.001 noise).

| 전략 | judge | MRR | vs baseline (MRR) |
|---|---:|---:|---:|
| **query_expansion** | **3.998** | 0.7783 | +0.0086 |
| hyde | 3.985 | 0.7738 | +0.0041 |
| decompose | 3.967 | 0.7786 | +0.0089 |
| query2doc | 3.967 | 0.7768 | +0.0071 |
| multi_query_para | 3.953 | 0.7780 | +0.0083 |
| baseline (no PreR) | 3.953 | 0.7697 | (기준) |
| step_back | 3.941 | 0.7692 | −0.0005 |

→ **Pre-retriever 단변량 효과는 사실상 noise 범위** (judge 3.94~4.00). query_expansion 이 judge 1위지만 baseline 과 +0.045 차에 불과. decompose 가 MRR 은 근소 우위(0.7786)이나 judge 는 하위 → 순위는 생성 품질(judge) 기준. step_back 같은 추상화는 오히려 약간 손해. 단답형 한국어 RAG에서 query는 이미 정확하므로 변형 효과 미미.

### 4.6 Stage 4-2: Post-retriever (RQ3)

25종 reranker 평가. Top 8:

| 순위 | 모델 | 파라미터 | MRR | Hit@1 |
|---:|---|---|---:|---:|
| 🥇 (tie) | dragonkue/bge-reranker-v2-m3-ko | 568M (KR fine-tune) | **0.7697** | **74.0%** |
| 🥇 (tie) | shoxa-mir/bge-reranker-v2-m3-ko | 568M | **0.7697** | **74.0%** |
| 🥈 | jinaai/jina-reranker-m0 | 2.4B 멀티모달 | 0.7631 | 72.3% |
| 🥉 | Qwen/Qwen3-Reranker-4B (2025 SOTA) | 4B | 0.7514 | 70.3% |
| 4 | BAAI/bge-reranker-v2-m3 | 568M multilingual | 0.7488 | 70.0% |
| 5 | SeoJHeasdw/ktds-vue-code-search-reranker-ko | 0.6B | 0.7439 | 70.0% |
| 6 | mxbai-rerank-base-v2 | 0.5B | 0.7373 | 68.3% |
| 7 | upskyy/ko-reranker-8k | 0.6B | 0.7343 | 68.0% |

→ **한국어 fine-tune이 절대 우세** (RQ3 답): KR-tune 0.7697 > Qwen3-4B 0.7514 (+1.89pp, 6.7배 작은 모델로). 11/25 (44%) 모델이 no_rerank baseline 미달.

### 4.7 Stage 5 axis-wise (시나리오 F)

28 configs × 300q × (1 gen + 4 judge) = 42,000 GPT-5.4 호출. Top winner per axis:

| Axis | Winner | judge |
|---|---|---:|
| A. PreR | query_expansion | 3.998 |
| B. R | Hybrid 5:5 | 3.983 |
| C. PostR | **Dongjin-kr/ko-reranker** | **4.005** |

**검색 1위와 생성 1위가 다르다**: retrieval에서 dragonkue/bge-v2-m3-ko가 1위(MRR 0.7697)였으나 generation 품질은 Dongjin-kr/ko-reranker가 1위(judge 4.005). 두 모델 모두 한국어 fine-tune 변종이지만 미세한 응답 분포 차이가 생성 품질에 다른 영향.

### 4.8 Stage 6 Cartesian (RQ2)

8 × 6 × 8 = 384 configs × 300q × (1 gen + 4 judge) = **576,000 GPT-5.4 호출**, 2.5h wallclock, ~$290.

**Top 10 (judge 기준)**:

| 순위 | PreR | R | PostR | MRR | judge |
|---:|---|---|---|---:|---:|
| 🥇 | **query2doc** | **hybrid_7_3** | **jina-reranker-m0** | 0.7630 | **4.067** |
| 🥈 | query_expansion | hybrid_5_5 | jina-reranker-m0 | 0.7806 | 4.062 |
| 🥉 | query2doc | hybrid_5_5 | jina-reranker-m0 | 0.7726 | 4.036 |
| 4 | query_expansion | dense | jina-reranker-m0 | 0.7607 | 4.032 |
| 5 | query2doc | hybrid_5_5 | ko-reranker | 0.7401 | 4.030 |

**MRR 최고**:

| 순위 | config | MRR | Hit@1 | judge |
|---:|---|---:|---:|---:|
| 🥇 | **multi_query_para + hybrid_5_5 + jina-reranker-m0** | **0.7874** | **75.0%** | 3.991 |
| 🥈 | multi_query_para + hybrid_ws_5_5 + dragonkue/bge-v2-m3-ko | 0.7850 | 75.0% | 4.019 |

**핵심 발견 (RQ2 답)**:

1. **jina-reranker-m0가 cartesian winner** — axis-wise 미실험이었던 멀티모달 reranker(Qwen2-VL 2.4B 백본)가 Top 10 중 8개를 차지하며 압도적 우세.
2. **query2doc × jina-m0 상호작용**: axis-wise PreR ranking에서 query2doc는 4위였으나 cartesian에서 jina-m0와 조합 시 1, 3, 5, 7, 8위 등장. 단변량 평가는 이 상호작용을 발견하지 못한다.
3. **컴포넌트 영향력 정량화**: judge 변동 폭으로 측정 시 PostR (Δ0.45, 12%) > R (Δ0.31, 8%) > PreR (Δ0.06, 1.5%). **Reranker 선택이 압도적 중요**.
4. **Bottom 10 전부 no_rerank** — 가장 안 좋은 component가 가장 큰 영향. Reranker 사용 자체가 필수.

### 4.9 누적 개선 분석 (RQ4)

가장 단순한 baseline (PreR=baseline + R=dense + PostR=no_rerank) 대비:

| 단계 | MRR | Δ MRR | Hit@1 | Δ Hit@1 | judge | Δ judge |
|---|---:|---:|---:|---:|---:|---:|
| 0. naive | 0.6816 | — | 59.0% | — | 3.850 | — |
| 1. + Hybrid (KIWI) | 0.7171 | +5.2% | 65.3% | +6.3pp | 3.869 | +0.5% |
| 2. + Reranker (dragonkue) | 0.7697 | +12.9% | 74.0% | +15.0pp | 3.916 | +1.7% |
| 3. axis-wise winner | 0.7747 | +13.7% | 70.7% | +11.7pp | 3.983 | +3.5% |
| **4. Cartesian (judge 최고)** | **0.7630** | +11.9% | 71.3% | +12.3pp | **4.067** | **+5.6%** |
| 🏆 **MRR 최고 조합** | **0.7874** | **+15.5%** | **75.0%** | **+16.0pp** | 3.991 | +3.7% |

→ 누적 최대 **MRR +15.5%, Hit@1 +27.1% (300문제 중 48문제 추가 정답), Judge +5.6%**.

### 4.11 Pipeline 최적화 vs 모델 업그레이드 (Phase 5 비교)

Cartesian과 Phase 5는 동일 dataset(allganize 300 Q&A), 동일 embedding (gemma-300m), 동일 평가 규칙(18-judge 4-metric majority O)을 사용. 따라서 accuracy 직접 비교 가능.

| Pipeline | Generator | Accuracy |
|---|---|---:|
| 🥇 **Cartesian winner**: query2doc + Hybrid 7:3 + jina-reranker-m0 | GPT-5.4 | **0.827** |
| Phase 5 단순 retrieval + GPT-5.4 | GPT-5.4 | 0.787 |
| Phase 5 + GPT-5.4-pro (10× 더 비싼 모델) | GPT-5.4-pro | 0.767 |
| Phase 5 Open Weights 1위 | gpt-oss_120b / kimi-k2.5 | 0.740 |

→ **동일 generator 모델(GPT-5.4)에서 RAG 파이프라인 최적화만으로 accuracy +4.0pp**. 더 큰 모델(GPT-5.4-pro)로 업그레이드한 것보다 **+6.0pp** 더 높음. 즉 본 데이터셋에서는 **모델 업그레이드 < RAG 파이프라인 최적화** 가성비.

### 4.10 Stage 5 Phase 5 (Open vs Closed Generation)

별도 trajectory로 46개 generation 모델 × 18 judge 모델 평가 (3-judge flagship: GPT-5.4-pro, Claude Opus 4.7, Gemini-3.1-pro).

| 카테고리 | 1위 모델 | Consolidated acc | Wilson 95% CI |
|---|---|---:|---:|
| Closed | gpt-5.4 | 0.787 | [0.737, 0.829] |
| Closed (target) | gpt-5.4-pro | 0.767 | [0.716, 0.811] |
| **Open Weights** | **gpt-oss_120b / kimi-k2.5 (tie)** | **0.740** | [0.688, 0.786] |

→ **Open Weights 진영에서 OpenAI gpt-oss와 Moonshot Kimi K2가 Closed 최상위에 가장 근접**. 5pp 이내 Open 후보 5개 모두 gpt-5.4-pro의 Wilson CI와 겹침(noise range 내). 자세한 self-bias 분석, 도메인별 분산, judge 일치도는 `docs/5_generation.md` 참조.

---

## 5. 토의

### 5.1 단변량 vs Cartesian의 가치

본 연구는 동일 데이터셋에서 두 가지 평가 방식을 모두 수행하여 그 차이를 정량화했다.

| 평가 | 1위 PreR | 1위 PostR | judge gain over naive |
|---|---|---|---:|
| Stage 4-1 단변량 (retrieval만) | query_expansion | (PostR axis 동결) | retrieval +0.18pp MRR |
| Stage 5 axis-wise (gen+judge) | query_expansion | ko-reranker | +3.1% |
| **Stage 6 Cartesian** | **query2doc** | **jina-reranker-m0** | **+5.6%** |

상호작용 효과는 단변량 1위 조합으로 도달 불가능하다. Cartesian 평가는 ~$300 추가 비용으로 추가 +2.1pp judge gain을 발견하며, 이는 단변량만으로는 보이지 않는다.

### 5.2 한국어 fine-tune의 일관된 가치

3개 컴포넌트에서 동일 패턴이 관찰됐다.

| 컴포넌트 | 한국어 fine-tune 1위 | 일반 SOTA 1위 | 차이 |
|---|---|---|---:|
| Embedding | KoE5 (1024d) | Qwen3-Embed-8B (4096d) | **+0.16 MRR** |
| BM25 tokenizer | KIWI 형태소 | whitespace | **+14.4pp MRR** |
| Reranker | dragonkue/bge-v2-m3-ko (568M) | Qwen3-Reranker-4B | **+1.89pp MRR** |

이는 한국어 NLP에서 **모델 크기·신선도 < 한국어 도메인 정렬**이라는 일반적 통찰과 부합한다.

### 5.3 검색 metric과 생성 품질의 괴리

cartesian에서 MRR 최고 조합과 judge 최고 조합이 다르다.

| 목적 | 권장 pipeline |
|---|---|
| 검색 정확도 최우선 | multi_query_para + Hybrid 5:5 + jinaai/jina-reranker-m0 (MRR 0.7874) |
| 답변 품질 최우선 | query2doc + hybrid_7_3 + jina-reranker-m0 (judge 4.067) |

이는 retrieval과 generation이 다른 metric 공간이며, 운영 목적에 따라 다른 winner를 선택해야 함을 시사한다. 단일 metric으로 최적화하면 다른 측면을 놓칠 수 있다.

### 5.4 의외의 결과

- **단순한 character-level recursive split이 LLM-기반 chunker를 능가** (Stage 2). 단답형 한국어 RAG에서 의미 경계는 character-level만큼 정확하지 않다.
- **mxbai-rerank base > large** (Stage 4-2): 1.5B 모델이 0.5B 모델보다 낮음 (0.6945 vs 0.7373). 크기 무조건 우수 가정 반례.
- **naver/xprovence-reranker catastrophic** (MRR 0.0496): 용도(context pruning)와 본 평가(reranking) 불일치로 사실상 무작위 결과.

### 5.5 한계

- **단답형 factoid Q&A 한정**: allganize 데이터셋은 정답이 명확한 단답형. 멀티홉, 추상 reasoning, 긴 서술 답변에는 결론 다를 수 있음.
- **공정성**: Stage 3 embedding 측정은 Stage 1·2 winner 확정 전 진행돼 parser=pymupdf4llm + 500/100 베이스라인 사용. 임베딩 간 상대 순위는 chunker 효과가 균등 작용한다고 가정 시 유효.
- **LLM-as-Judge 편향**: GPT-5.4 단일 judge로 cartesian 측정. Phase 5에서는 3-judge ensemble로 검증했으나 cartesian은 단일.
- **transformers v5 호환성 실패 reranker 7종**: jina-reranker-v3, gte-multilingual, sigridjineth, bge-v2-minicpm 등 평가 누락. 향후 환경 핀 후 재실험 가치 있음.

---

## 6. 결론

본 연구는 한국어 RAG 파이프라인의 컴포넌트별 효과를 단변량 + cartesian 두 단계로 분해 평가했다. 95+ 구성 요소 비교와 580K+ LLM-as-Judge 호출을 통해 다음을 실증했다.

1. **RQ1**: 컴포넌트 영향력은 PostR > R > PreR 순서. Reranker가 압도적 가장 중요 (judge 변동 폭 12%).
2. **RQ2**: Cartesian 평가는 단변량 대비 +2.1pp judge gain을 추가 발견 (query2doc × jina-m0 상호작용).
3. **RQ3**: 한국어 fine-tune이 일반 multilingual SOTA를 일관되게 능가 (embedding, BM25, reranker 모두).
4. **RQ4**: 단순 dense baseline → 최적 cartesian: 누적 MRR +15.5%, Hit@1 +27.1%, judge +5.6%.

**권장 운영 pipeline**:

```
PyMuPDFLoader
  → RecursiveCharacterTextSplitter(300, 50)
  → google/embeddinggemma-300m
  → query2doc (PreR, GPT-5.4 가상문서 생성)
  → Hybrid 7:3 (FAISS Dense + BM25-KIWI, RRF k=60)
  → top-20
  → jinaai/jina-reranker-m0
  → top-5
  → GPT-5.4 RAG 답변
```

**전체 실험 코드, 데이터, 결과**는 [github.com/BAEM1N/RAG-Evaluation](https://github.com/BAEM1N/RAG-Evaluation) (MIT License)와 [HuggingFace dataset](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark)으로 공개한다.

향후 연구로는 (i) 멀티홉 / 긴 답변 데이터셋으로의 확장, (ii) LLM-as-Judge ensemble을 Cartesian에 적용, (iii) Stage 4-2 호환성 실패 reranker 7종 재실험, (iv) Korean fine-tune 신규 모델 (Qwen3-Reranker-8B 등)의 지속 추적이 있다.

---

## 부록 A. 비용 분석

전체 실험 ~$554 (LLM API). 로컬 GPU 추론 ~15h ($0).

| 단계 | 호출 수 | 시간 | 비용 |
|---|---:|---:|---:|
| Stage 1-4 단변량 (로컬) | 0 | 8h | $0 |
| Stage 4-1 PreR (GPT-5.4) | 2,400 | 5분 | ~$3 |
| Stage 5 axis-wise (GPT-5.4) | 42,000 | 2h | ~$21 |
| Stage 6 Cartesian (GPT-5.4) | 576,000 | 2.5h | ~$290 |
| Stage 5 Phase 5 (multi-vendor) | ~240,000 | 48h batch | ~$240 |
| **합계** | **~860K** | — | **~$554** |

자세한 단가, 시나리오별 추정은 [`docs/cost-report.md`](./cost-report.md) 참조.

## 부록 B. 인프라

| 장비 | 사양 | 용도 |
|---|---|---|
| MacBook Pro 14 | M5 Max, 128GB | 메인 driver, 임베딩, reranker (M5 MPS) |
| HP Z2 Mini G1a | AMD Ryzen AI Max+ 395, 128GB | Reranker (AI 395+ ROCm 7.13) |
| DGX Spark | GB10, 128GB | Phase 5 local LLM (ollama) |
| OpenAI OpenAI-compatible API | gpt-5.4 deployment (10M TPM / 100K RPM) | Stage 4-1/5/6 LLM 호출 |
| Anthropic / OpenAI / Google / OpenRouter | 다중 vendor | Phase 5 19 Closed + 3-judge |

## 부록 C. 참고문헌

### 핵심 메서드
- Gao et al. 2023. "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE). ACL. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- Wang et al. 2023. "Query2doc: Query Expansion with Large Language Models". EMNLP. [arXiv:2303.07678](https://arxiv.org/abs/2303.07678)
- Zheng et al. 2024. "Take a Step Back: Evoking Reasoning via Abstraction". ICLR. [arXiv:2310.06117](https://arxiv.org/abs/2310.06117)
- Press et al. 2023. "Measuring and Narrowing the Compositionality Gap" (Self-Ask). EMNLP. [arXiv:2210.03350](https://arxiv.org/abs/2210.03350)
- Jagerman et al. 2023. "Query Expansion by Prompting LLMs". [arXiv:2305.03653](https://arxiv.org/abs/2305.03653)
- Ma et al. 2023. "Query Rewriting for Retrieval-Augmented LLMs". EMNLP. [arXiv:2305.14283](https://arxiv.org/abs/2305.14283)
- Robertson & Zaragoza 2009. "The Probabilistic Relevance Framework: BM25 and Beyond". FnT IR. [DOI](https://doi.org/10.1561/1500000019)
- Cormack et al. 2009. "Reciprocal Rank Fusion outperforms Condorcet" (RRF). SIGIR. [DOI](https://doi.org/10.1145/1571941.1572114)
- Chen et al. 2024. "BGE M3-Embedding". [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- Liu et al. 2023. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment". [arXiv:2303.16634](https://arxiv.org/abs/2303.16634)
- Lewis et al. 2020. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". NeurIPS. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

### 데이터셋 / 라이브러리
- allganize/RAG-Evaluation-Dataset-KO: [HuggingFace](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
- LangChain: [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- Chonkie: [github.com/chonkie-inc/chonkie](https://github.com/chonkie-inc/chonkie)
- Kiwi 한국어 형태소: [github.com/bab2min/Kiwi](https://github.com/bab2min/Kiwi)

### 모델 (HuggingFace)
- KoE5: [nlpai-lab/KoE5](https://huggingface.co/nlpai-lab/KoE5)
- EmbeddingGemma 300m: [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
- dragonkue/bge-reranker-v2-m3-ko: [link](https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko)
- jinaai/jina-reranker-m0: [link](https://huggingface.co/jinaai/jina-reranker-m0)
- Qwen/Qwen3-Reranker-4B: [link](https://huggingface.co/Qwen/Qwen3-Reranker-4B)

(전체 95+ 모델 카탈로그: [`docs/model-inventory.md`](./model-inventory.md))

---

## 인용

```bibtex
@misc{baem1n2026koragbench,
  title  = {한국어 RAG 파이프라인의 컴포넌트별 효과 분해 — 단변량과 Cartesian 분석을 통한 실증 연구},
  author = {BAEM1N},
  year   = {2026},
  url    = {https://github.com/BAEM1N/RAG-Evaluation}
}
```
