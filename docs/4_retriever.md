# Retriever — Dense / Sparse / Hybrid 비교

> **데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
>
> **고정 조건**: loader=pymupdf, parser=LC Recursive 300/50, embedding=embeddinggemma-300m, top-k=5
>
> **chunks**: 4,556 (Stage 2 winner와 동일)

검색 방식만 변경. RRF (Reciprocal Rank Fusion) k=60으로 hybrid 결합.

## 1. 7종 비교 결과

| 전략 | MRR | Hit@1 | Hit@5 | File@5 |
|---|---:|---:|---:|---:|
| **Hybrid 3:7 (dense + bm25-KIWI)** | **0.7171** | **65.3%** | 80.3% | **91.7%** |
| Hybrid 5:5 (dense + bm25-KIWI) | 0.7137 | 65.3% | 80.0% | 91.7% |
| Hybrid 7:3 (dense + bm25-KIWI) | 0.7046 | 64.0% | 80.3% | 91.7% |
| Dense (embeddinggemma-300m) | 0.6816 | 59.0% | **81.3%** | 91.7% |
| BM25 + KIWI | 0.6783 | 61.3% | 77.3% | 89.3% |
| Hybrid 5:5 (dense + bm25-whitespace) | 0.6496 | 59.0% | 74.3% | 88.0% |
| BM25 + whitespace | 0.5344 | 48.3% | 62.7% | 77.7% |

**선택: `Hybrid 3:7 (dense + BM25-KIWI)`** — Dense 단독 대비 MRR +3.55pp, Hit@1 +6.3pp.

## 2. 핵심 관찰

### (1) Hybrid가 단독 검색을 모두 이긴다

세 hybrid 비율(7:3, 5:5, 3:7) 모두 dense 단독(0.6816)과 BM25-KIWI 단독(0.6783)을 능가. 두 방식이 서로 다른 실수를 잡아내고 있다는 의미.

| 방식 | MRR | vs Hybrid 3:7 |
|---|---:|---:|
| Hybrid 3:7 | 0.7171 | — |
| Dense alone | 0.6816 | −3.55pp |
| BM25-KIWI alone | 0.6783 | −3.88pp |

### (2) BM25는 KIWI(형태소 분석)가 필수

| BM25 토크나이저 | MRR | Hit@5 |
|---|---:|---:|
| KIWI (형태소) | 0.6783 | 77.3% |
| 공백 분리 | 0.5344 | 62.7% |
| **차이** | **+14.39pp** | **+14.6pp** |

한국어는 조사/어미가 붙어 공백 split만으로는 검색 토큰이 단순화되지 않음. 형태소 분석 필수.

### (3) Dense ≈ BM25-KIWI — 단독 성능 거의 동률

- Dense MRR 0.6816 vs BM25-KIWI 0.6783, 차이 0.33pp.
- 한국어 RAG에서 **의미 검색과 키워드 검색이 동등**하다는 점이 흥미. embedding 모델만으로는 잡지 못하는 정확한 매칭이 BM25에는 있다는 뜻.
- 그래서 hybrid 비율이 BM25 쪽으로 약간 기울 때(3:7) 가장 좋음.

### (4) hybrid weight sweet spot — 3:7 또는 5:5

| 비율 (dense:BM25-KIWI) | MRR |
|---|---:|
| 7:3 | 0.7046 |
| 5:5 | 0.7137 |
| 3:7 | **0.7171** |

BM25 쪽 가중치를 높일수록 약간 더 좋지만 격차는 작음(±1pp). 운영상으로는 5:5로 무난.

### (5) Hit@5 만큼은 Dense가 약간 우세

- Dense Hit@5 = 81.3% > Hybrid 3:7 = 80.3%
- 1위 정확도는 hybrid가 압도(65.3% vs 59.0%)지만, top-5 안에 들어오는 비율은 dense가 살짝 높음.
- LLM이 top-5 컨텍스트로 답하는 형태(현재 Phase 5 설정)에서는 둘 다 비슷한 효과지만, top-1만 쓰는 시스템이면 hybrid 우위가 결정적.

### (6) 잘못된 hybrid — BM25-whitespace + Dense는 Dense보다 낮다

- Hybrid 5:5 (dense + bm25-whitespace) = 0.6496 < Dense alone = 0.6816
- 약한 sparse retriever를 합치면 오히려 노이즈가 들어가 점수가 깎임.
- **교훈**: hybrid는 두 retriever 모두 단독으로도 어느 정도 강해야 효과가 있음.

## 3. RRF 공식

$$ \text{score}(c) = \sum_i \frac{w_i}{k + \text{rank}_i(c)} $$

- $k = 60$ (관례)
- $w_i$ = 각 retriever 가중치 (예: 0.3 dense + 0.7 BM25-KIWI)
- $\text{rank}_i(c)$ = retriever $i$에서 chunk $c$의 순위 (1부터)

## 4. 다음 단계 고정 값

- **Loader**: `pymupdf`
- **Parser**: `RecursiveCharacterTextSplitter(300, 50)`
- **Embedding**: `google/embeddinggemma-300m`
- **Retriever**: **Hybrid RRF 3:7 (dense + BM25-KIWI), k=60**

## 5. 메모

- 이전 docs의 7개 vector store 비교(FAISS/LanceDB/Qdrant/Milvus/Weaviate/Chroma/pgvector)는 별도 결론: 모든 store가 동일 embedding/chunk 입력에 대해 사실상 같은 top-5를 반환하므로 정확도 차이는 noise 수준이고 latency/운영성으로 결정. 본 문서는 retrieval **방식** (Dense/Sparse/Hybrid) 비교에 집중.
- Cross-encoder reranker, query 변형 등 retrieval 전/후 처리는 별도 문서:
  - [4-1_pre_retriever.md](./4-1_pre_retriever.md): HyDE, multi-query, query expansion
  - [4-2_post_retriever.md](./4-2_post_retriever.md): cross-encoder reranker, filtering
