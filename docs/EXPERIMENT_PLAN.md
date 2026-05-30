# RAG 단변량 벤치마크 — 향후 실험 계획

> 작성 기준일: 2026-05-20
> 데이터셋: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
> 고정 winner pipeline: `pymupdf → LC Recursive 300/50 → embeddinggemma-300m → Hybrid RRF 3:7 (dense + BM25-KIWI) → bge-reranker-v2-m3-ko`

---

## 0. 현재 상태

| Stage | 상태 | Winner | MRR |
|---|---|---|---:|
| 1 Loader | ✅ done | pymupdf | 0.6486 |
| 2 Parser (chunk size + char-based) | ✅ done | LC Recursive 300/50 | 0.6816 |
| 2-ext Parser (semantic chunkers) | ⏳ **미실험** | — | — |
| 3 Embedding | ✅ done (27 모델 leaderboard) | KoE5 / gemma-embed-300m | 0.6871 |
| 4 Retriever | ✅ done | Hybrid 3:7 (Dense + BM25-KIWI) | 0.7171 |
| 4-1 Pre-retriever | 🔄 진행 중 (10 전략 × 300q, Gemini Flash Lite) | TBD | TBD |
| 4-2 Post-retriever | 🟡 부분완료 (3개) — **신모델 추가 필요** | bge-reranker-v2-m3-ko | 0.7697 |
| 5 Generation | ✅ done (Phase 5, 46 모델 × 18 judges) | gpt-5.4-pro | — |

---

## 1. Stage 2-ext: Semantic Chunker 실험 (LLM 무비용)

`2_parser.md` 의 char-based chunker 12종 비교에서 **semantic 계열 5종이 누락**. 추가 embedding 호출 비용은 발생하나 LLM 비용은 0.

### 비교 후보 7종

| 전략 | 라이브러리 | 동작 | 추가 비용 |
|---|---|---|---|
| **LC SemanticChunker** | `langchain-experimental` | embedding 유사도 변화점에서 분할 (percentile/standard_deviation/IQR threshold) | embedding 1회/chunk 후보 |
| **LC MarkdownHeaderTextSplitter** | `langchain_text_splitters` | `#`/`##` 헤더 단서로 구조 분할 | 0 |
| **Chonkie SemanticChunker** | `chonkie` | embedding 의미 묶기 (similarity threshold) | embedding |
| **Chonkie SDPMChunker** | `chonkie` | Semantic Double-Pass Merging — semantic 묶기 후 재병합 | embedding × 2 |
| **Chonkie NeuralChunker** | `chonkie` | 학습된 chunking 모델 (BERT classifier) | 모델 로딩 |
| **LlamaIndex SemanticSplitterNodeParser** | `llama-index-core` | embedding 의미 기반, breakpoint percentile | embedding |
| **Unstructured chunk_by_title** | `unstructured` | 제목 단위 청킹 (loader Unstructured와 페어링) | 0 |
| **KSS + Recursive (자체)** | `kss` + `langchain` | 한국어 문장 분리기 KSS 적용 후 재귀 분할 | 0 |

### 측정 변수

고정: loader=pymupdf, embedding=gemma-embed-300m, retriever=Hybrid 3:7, top-k=5

가변: chunker, chunk size 파라미터

### 비용 / 시간

- embedding 횟수: 4,556 chunks × 7 모델 = 31,892 추가 호출 (gemma-embed-300m 로컬, 무료)
- LLM 호출: 0
- 시간: 약 30~45분 (7 strategy × ~5min embedding)

### 실행 명령 (계획)

```bash
python scripts/bench_parser_semantic.py --strategies all
```

스크립트는 `bench_parser_extended.py` 를 베이스로 semantic 계열 8종 추가.

---

## 2. Stage 4-2 확장: 최신 Reranker (2025-2026)

현재 3개(no_rerank, bge-v2-m3, bge-v2-m3-ko)만 측정. 사용자 지적: "너무 옛날 거만 쓴다".
2025년 이후 출시된 7개 + 비-reranker 전략 4개 추가 예정.

### 2A. 신규 Reranker 모델 (7개)

| 모델 | 파라미터 | 출시 | KR | HF URL |
|---|---:|---|---|---|
| `Qwen/Qwen3-Reranker-0.6B` | 0.6B | 2025-06 | O (100+) | [link](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| `Qwen/Qwen3-Reranker-4B` | 4B | 2025-06 | O | [link](https://huggingface.co/Qwen/Qwen3-Reranker-4B) |
| `jinaai/jina-reranker-v3` | 0.6B | 2025-10 | O | [link](https://huggingface.co/jinaai/jina-reranker-v3) |
| `mixedbread-ai/mxbai-rerank-base-v2` | 0.5B | 2025 | O | [link](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) |
| `mixedbread-ai/mxbai-rerank-large-v2` | 1.5B | 2025 | O | [link](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) |
| `BAAI/bge-reranker-v2.5-gemma2-lightweight` | ~2B | 2024-Q4 | O | [link](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight) |
| `Dongjin-kr/ko-reranker` | bge fine-tune | 2024-2025 | O (KR) | [link](https://huggingface.co/Dongjin-kr/ko-reranker) |
| `naver/modernReranker` | ModernBERT-L | 2025 | 부분 | [link](https://huggingface.co/naver/modernReranker) |
| `telepix/PIXIE-Spell-Reranker-Preview-0.6B` | 0.6B | 2025 | O (KR/EN) | [link](https://huggingface.co/telepix/PIXIE-Spell-Reranker-Preview-0.6B) |

### 2B. Reranker 외 Post-retrieval 전략 (4개)

| 전략 | 메모 | 레퍼런스 |
|---|---|---|
| **MMR** (Maximum Marginal Relevance) | 다양성 + 관련성 균형. λ ∈ {0.3, 0.5, 0.7} 튜닝 | Carbonell & Goldstein 1998 SIGIR |
| **Multi-reranker RRF** | 2개 reranker 결과 RRF 앙상블 | Cormack 2009 |
| **Score normalization + rerank** | min-max / percentile 보정 후 재정렬 | zELO 2509.12541 |
| **Sentence-level rerank (DSLR)** | chunk → 문장 분해 → 무관 문장 필터 | 2407.03627 |

### 2C. 향후 (LLM 비용 발생, 별도 예산)

| 전략 | 비용 | 메모 |
|---|---|---|
| LLM-as-reranker (RankGPT/RankZephyr/RankLLaMA) | GPT-5.4 약 $0.5/실험 | top-N 전체 listwise 재정렬 |
| LongLLMLingua / LLMLingua-2 (contextual compression) | 압축 모델 무료 + LLM 호출 | chunk 압축 후 LLM 컨텍스트 단축 |
| ChunkRAG | LLM chunk 필터링 | 무관 chunk LLM 제거 |
| Cohere rerank-v4.0 (API) | $1/1k 검색 × 300 = $0.30 | 클로즈드 |
| Voyage rerank-2.5 (API) | $0.05/1M tok | 클로즈드 |

### 비용 / 시간

- 로컬 reranker 9개: M3 Max 추정 100~150초 × 9 = **~20분**
- LLM-based 전략: 별도 시나리오에서 처리

### 실행 명령

```bash
# 기존 스크립트 모델 리스트만 확장
python scripts/bench_reranker.py --rerankers all
```

---

## 3. End-to-End 생성/평가 시나리오 (GPT-5.4)

`docs/cost-report.md` 실측 토큰값 기반.

### 단위 비용

| 항목 | 토큰 | 단가 (batch) | 비용 |
|---|---|---|---:|
| 생성 input (923 × 300) | 277K | $1.25/M | $0.35 |
| 생성 output (100 × 300) | 30K | $7.50/M | $0.23 |
| **생성 소계** | | | **$0.58** |
| Judge input (385 × 4 metric × 300) | 462K | $1.25/M | $0.58 |
| Judge cache 적용 (66% 정적 × 50% off) | | | −$0.19 |
| Judge output (1 × 4 × 300) | 1.2K | $7.50/M | $0.01 |
| **Judge 소계 (cached)** | | | **~$0.40** |
| **Per config 합계** | | | **~$0.98** |

### 시나리오

| 시나리오 | 변수 | configs | 비용 |
|---|---|---:|---:|
| **A. PreR 단변량** | 10 PreR × R-winner × PostR-winner | 10 | **$10** |
| **B. R 단변량** | PreR-winner × 7 R × PostR-winner | 7 | **$7** |
| **C. PostR 단변량** | PreR-winner × R-winner × 12 PostR (3 기존 + 7 new + 2 strategy) | 12 | **$12** |
| **D. 선별 cartesian** | PreR top-3 × R top-3 × PostR top-3 | 27 | **$27** |
| **E. 전체 cartesian** | 10 × 7 × 12 | 840 | ~$820 |
| **F. End-to-end pipeline survey** | A+B+C 합산 | 29 | **$29** |

### 권장 실행 순서

```
1단계 [검색metric만, 무료]: Stage 4-1 완료 (현재 진행) + Stage 4-2 신모델 추가
  → PreR/R/PostR top-3 winner 식별

2단계 [생성+평가, ~$29]: 시나리오 F (axis-wise A+B+C)
  → 각 axis별 생성품질 영향 정량화

3단계 [선별 cartesian, ~$27]: 시나리오 D
  → PreR×R×PostR 상호작용 효과 측정 (top-3씩만)

4단계 [선택]: 전체 cartesian E ($820) 또는 GPT-5.4-pro pro ($2,100)
```

**합계 권장**: 2단계 + 3단계 = **~$56** + 20% 버퍼 = **~$70**

### 주의

- Judge 출력 1토큰 유지 필수 (reasoning judge로 키우면 +50% 비용)
- 동일 question에 대한 candidate들을 contiguous batch 처리해 cache TTL 5분 안에 hit 극대화
- OpenRouter 우회 시 markup 5-10% 추가 — Direct API 사용 권장
- 메모리의 `[[feedback_api_eval_lessons]]` 원칙 준수: 1-call dry-run → 100-call pilot → full 300

---

## 4. 전체 진행 우선순위

| Phase | 작업 | 비용 | 시간 |
|---|---|---:|---:|
| **P1** (즉시) | Stage 4-1 완료 대기 (running) | $0 | ~60min |
| **P2** | Stage 4-2 신모델 9개 + MMR + multi-rerank RRF 추가 | $0 | ~30min |
| **P3** | Stage 2-ext Semantic chunker 7종 | $0 | ~45min |
| **P4** | 시나리오 F (axis-wise gen+eval) | $29 | ~24h batch |
| **P5** | 시나리오 D (선별 cartesian) | $27 | ~24h batch |
| **P6** (선택) | LLM-based post-retriever (RankGPT, Lingua) | ~$15 | ~24h |
| **P7** (선택) | 전체 cartesian E | $820 | ~48-72h |

**핵심 의사결정 시점**:
- P1+P2+P3 완료 후 retrieval winner 재확정
- P4 결과로 생성품질이 검색metric과 얼마나 상관/괴리하는지 정량화
- P5 결과로 PreR/R/PostR 간 시너지 (특히 PreR + Reranker 결합 효과) 확인

---

## 5. 산출물

각 phase 완료 시:

| 문서 | 갱신 내용 |
|---|---|
| `docs/2_parser.md` | semantic chunker 8종 결과 추가 |
| `docs/4-1_pre_retriever.md` | 10 PreR 전략 신규 작성 (현재 미작성) |
| `docs/4-2_post_retriever.md` | 12 PostR 전략 + 신모델 9종 갱신 |
| `docs/5_generation.md` | axis-wise 생성품질 영향 + cartesian 결과 |
| `docs/REPORT.md` | end-to-end winner pipeline 갱신 |
| HuggingFace dataset | parquet에 신 strategy 결과 추가 |

---

## 6. 레퍼런스 (각 stage docs에 인용 예정)

### Pre-retriever
- HyDE — Gao et al. 2022, [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- Query2doc — Wang et al. 2023 EMNLP, [arXiv:2303.07678](https://arxiv.org/abs/2303.07678)
- Step-back prompting — Zheng et al. 2023 ICLR, [arXiv:2310.06117](https://arxiv.org/abs/2310.06117)
- Self-Ask — Press et al. 2022, [arXiv:2210.03350](https://arxiv.org/abs/2210.03350)
- Query Expansion (LLM) — Jagerman et al. 2023, [arXiv:2305.03653](https://arxiv.org/abs/2305.03653)
- Query Rewriting — Ma et al. 2023 EMNLP, [arXiv:2305.14283](https://arxiv.org/abs/2305.14283)
- Multi-query (LangChain) — [docs](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)

### Retrieval
- BM25 — Robertson & Zaragoza 2009 FnT IR, [DOI](https://doi.org/10.1561/1500000019)
- RRF — Cormack et al. 2009 SIGIR, [DOI](https://doi.org/10.1145/1571941.1572114)
- Kiwi 한국어 형태소 — [github.com/bab2min/Kiwi](https://github.com/bab2min/Kiwi)

### Post-retriever
- ColBERTv2 — [arXiv:2112.01488](https://arxiv.org/abs/2112.01488)
- PLAID — [arXiv:2205.09707](https://arxiv.org/abs/2205.09707)
- RankZephyr — [arXiv:2312.02724](https://arxiv.org/abs/2312.02724)
- RankGPT — [arXiv:2304.09542](https://arxiv.org/abs/2304.09542)
- MMR — Carbonell & Goldstein 1998 SIGIR ACL Anthology X98-1025
- LongLLMLingua — [arXiv:2310.06839](https://arxiv.org/abs/2310.06839)
- LLMLingua-2 — [arXiv:2403.12968](https://arxiv.org/abs/2403.12968)
- DSLR (sentence rerank) — [arXiv:2407.03627](https://arxiv.org/abs/2407.03627)
- ChunkRAG — [arXiv:2410.19572](https://arxiv.org/abs/2410.19572)
- BGE-M3 — [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- zELO score calibration — [arXiv:2509.12541](https://arxiv.org/abs/2509.12541)

### Models (HuggingFace)
- 27 embedding 모델: `docs/3_embedding.md` 참조
- 9 reranker 모델: 위 §2A 표 참조
- EmbeddingGemma 300m — [HF](https://huggingface.co/google/embeddinggemma-300m) · [docs](https://ai.google.dev/gemma/docs/embeddinggemma)

### 라이브러리
- pymupdf — [docs](https://pymupdf.readthedocs.io/) · [github](https://github.com/pymupdf/PyMuPDF)
- LangChain Text Splitters — [docs](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
- Chonkie — [github](https://github.com/chonkie-inc/chonkie)
- LlamaIndex SentenceSplitter — [docs](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/)
