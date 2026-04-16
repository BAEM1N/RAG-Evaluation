# RAG 벤치마크 실험 계획서

> 기반 데이터셋: [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO) (300 Q&A, 5도메인)  
> 원본 PDF: 외장 SSD에 보유  
> 작성일: 2026-04-09

---

## 1. 인프라

| 서버 | IP | 역할 | 사양 |
|------|-----|------|------|
| AI-395 | 192.168.50.245 | 임베딩/LLM 모델 서빙 | MI100 96GB VRAM |
| Mac Mini | 192.168.50.241 | 실험 스크립트, OSIRIS/THOTH | M2 16GB |
| **T7910** | **192.168.50.250** | **벡터스토어 서빙** | Dual Xeon 72T, 128GB RAM, Docker |

T7910 현재 상태: Qdrant, PostgreSQL, OpenSearch, Neo4j, Redis 이미 가동 중.

---

## 2. 원본 실험 분석

### allganize LangChain Tutorial (공식 Colab baseline)

출처: [HuggingFace 데이터셋 페이지의 LangChain Tutorial Colab](https://colab.research.google.com/drive/1Jlzs8ZqFOqqIBBT2T5XGBhr23XxEsvHb)  
사본: `references/allganize_langchain_tutorial.ipynb`

```python
# 원본 baseline (실제 코드 2개 셀 중 핵심)
loader = PyPDFLoader(pdf_path)
docs = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 6})

llm = ChatOpenAI(model="gpt-4-turbo")
prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

**정확한 baseline 구성:**

| 컴포넌트 | 구성 |
|---------|------|
| Parser | `PyPDFLoader` (load_and_split) |
| Chunking | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)` |
| Embedding | `OpenAIEmbeddings()` = text-embedding-ada-002 |
| VectorStore | **Chroma** (in-memory, 인덱스 기본) |
| Retriever | similarity, **k=6** |
| LLM | gpt-4-turbo |
| Prompt | LangChain Hub `rlm/rag-prompt` |

**리더보드 결과:** 61.0% (183/300) — GPT-4-Turbo 기준

### 우리 확장점 (기존 연구 대비)

| 항목 | allganize | AutoRAG(Marker) | ssisOneTeam | **우리** |
|------|-----------|-----------------|-------------|---------|
| 데이터 | 300 Q&A | 미명시 | 106 Q&A | 300 Q&A |
| Parser | **PyPDFLoader** 고정 | 미명시 | 미명시 | **3종 비교** |
| Chunking | **RecursiveSplit 1000/200 고정** | 미명시 | 미명시 | **4종 비교** |
| Embedding | **OpenAI ada-002** 1종 | 16종 (API 혼합) | 24종 (API 혼합) | **28종 (전부 로컬 GGUF)** |
| VectorStore | **Chroma** 고정 | 미비교 | 미비교 | **7종 비교** |
| Retriever k | **6** 고정 | 미명시 | 미명시 | k=1/3/5/10 분석 |
| Reranker | 없음 | 없음 | 없음 | (추후) |
| LLM | **gpt-4-turbo** (상용) | 미사용 | 미사용 | **로컬 오픈소스 → RunPod 확장** |
| 인프라 | 100% OpenAI API | 상용 API | 상용 API | **100% 로컬** |

---

## 3. 실험 Phase

### Phase 0: 데이터 준비

**작업:**
1. HuggingFace 평가 데이터셋 다운로드 (300 Q&A parquet)
2. 외장 SSD에서 원본 PDF 63개 → 실험 디렉토리 복사
3. Ground Truth 매핑: `{question, target_answer, target_file_name, target_page_no, context_type, domain}`
4. T7910에 벡터스토어 7종 Docker 준비
5. 실험 스크립트 작성

**디렉토리 구조:**
```
experiments/rag-bench/
├── data/
│   ├── pdfs/{finance,public,medical,law,commerce}/
│   ├── eval_dataset.parquet
│   └── ground_truth.json
├── results/
│   ├── phase1_parser/
│   ├── phase2_chunking/
│   ├── phase3_vectorstore/
│   ├── phase4_embedding/
│   └── phase5_llm/
├── scripts/
│   ├── bench_parser.py
│   ├── bench_chunking.py
│   ├── bench_vectorstore.py
│   ├── bench_embedding.py
│   ├── bench_llm.py
│   └── eval_utils.py
└── docker-compose.vectorstores.yml   # T7910용
```

---

### Phase 1: Parser 비교

**고정:** Chunking=1000/200, Embedding=Qwen3-Embed-8B, VectorStore=pgvector  
**변수:** PyPDF, pymupdf4llm, docling

| 측정 | 내용 |
|------|------|
| page_hit@1/5/10 | 정답 페이지의 청크가 검색되는지 |
| file_hit@1/5 | 정답 파일의 청크가 검색되는지 |
| MRR | 정답 청크의 평균 역순위 |
| context_type별 | paragraph / table / image 분리 분석 |
| domain별 | 금융/공공/의료/법률/상거래 |

**예상:** 3 × 300 = 900 검색, ~30분

---

### Phase 2: Chunking 전략 비교

**고정:** Phase 1 최적 Parser, Embedding=Qwen3-Embed-8B, VectorStore=pgvector  
**변수:**

| 전략 | chunk_size | overlap |
|------|-----------|---------|
| small | 500 | 100 |
| baseline | 1,000 | 200 |
| medium | 1,500 | 200 |
| large | 2,000 | 300 |

**예상:** 4 × 300 = 1,200 검색, ~30분

---

### Phase 3: VectorStore 비교 ⭐

**서버:** T7910 (192.168.50.250)  
**고정:** Phase 1~2 최적, Embedding=Qwen3-Embed-8B  
**변수:**

| # | VectorStore | 인덱스 | 설치 | T7910 현황 |
|---|------------|--------|------|-----------|
| 1 | **pgvector** | HNSW, IVFFlat | PostgreSQL 확장 | postgres:16 이미 가동 |
| 2 | **FAISS** | IVF, HNSW, PQ | pip (서버리스) | 신규 |
| 3 | **Chroma** | HNSW | pip 또는 Docker | 신규 |
| 4 | **Milvus** | IVF_FLAT, HNSW, DiskANN | Docker | 신규 |
| 5 | **Qdrant** | HNSW | Docker | **이미 가동 중** |
| 6 | **Weaviate** | HNSW | Docker | 신규 |
| 7 | **LanceDB** | IVF_PQ | pip (서버리스) | 신규 |

**Docker Compose (T7910 신규 추가분):**
```yaml
# docker-compose.vectorstores.yml (T7910)
services:
  chroma:
    image: chromadb/chroma:latest
    ports: ["8100:8000"]
  milvus-standalone:
    image: milvusdb/milvus:latest
    ports: ["19530:19530"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:latest
    ports: ["8101:8080"]
    environment:
      PERSISTENCE_DATA_PATH: /var/lib/weaviate
```

**절차:**
```
embeddings = embed_all_chunks_once()  # AI-395에서 1회 벡터화

for vectorstore in stores:
    1. T7910에서 벡터스토어 시작
    2. 벡터 데이터 삽입 → 인덱스 빌드 시간 측정
    3. 300개 질문 벡터 → T7910으로 검색 요청
    4. 정확도 + 속도 측정
```

**측정:**

| 메트릭 | 설명 |
|--------|------|
| Hit@1/5/10, MRR, NDCG@5 | 검색 정확도 |
| insert_time | 전체 청크 삽입 소요 (초) |
| index_build_time | 인덱스 빌드 소요 (초) |
| query_latency_avg | 평균 검색 지연 (ms) |
| query_latency_p95 | P95 검색 지연 (ms) |
| queries_per_sec | 초당 검색 처리량 |
| memory_usage | 메모리 사용량 |

**인덱스 튜닝 서브 실험 (상위 3개 스토어):**
- HNSW: ef_construction=100/200/400, M=16/32/48
- IVF: nlist=64/128/256, nprobe=8/16/32

**예상:** 7 스토어 × 300 질문 = 2,100 검색, ~1시간

---

### Phase 4: Embedding 모델 비교 ⭐

**서버:** AI-395 (모델 서빙), T7910 (Phase 3 최적 벡터스토어)  
**고정:** Phase 1~3 최적  
**변수:** 28개 임베딩 모델

**그룹별 VRAM 스케줄:**

| 그룹 | 모델 | VRAM | 진행 방식 |
|------|------|------|----------|
| A. 소형 (19개) | snowflake-ko, pixie-rune, kure, koe5, qwen3-0.6b, bge-m3, me5, jina-v5-small, harrier-0.6b, labse, nomic-moe, mxbai, voyage-nano, gemma-300m, granite-278m, harrier-270m, jina-v5-nano, granite-107m, snowflake-l-v2 | 각 0.2~1GB | 순차 교체 (빠름) |
| B. 중형 (4개) | qwen3-4b, jina-v4-retrieval, jina-v4-code, jina-code-1.5b | 각 2~4GB | 순차 교체 |
| C. 대형 (4개) | qwen3-8b, llama-embed-nemotron-8b, nemotron-embed-8b, e5-mistral-7b | 각 7~9GB | LLM 내리고 테스트 |
| D. 초대형 (1개) | harrier-27b | 20GB | 단독 실행 |

**절차:**
```
for model in embedding_models:
    1. 게이트웨이: 이전 모델 언로드 → 새 모델 로드
    2. 모든 청크 벡터화 (dim 기록)
    3. T7910 벡터스토어에 삽입
    4. 300개 질문 검색 → 메트릭
    5. 벡터화 속도 (chunks/sec) 기록
```

**예상:** 28 × 300 = 8,400 검색, ~4~6시간 (모델 로딩 포함)

---

### Phase 5: LLM 생성 비교

**고정:** Phase 1~4 최적 (전체 파이프라인)  
**변수:**

| # | LLM | 크기 | 위치 | 비고 |
|---|-----|------|------|------|
| 1 | Qwen3.5-27B | 26GB | AI-395 로컬 | 현재 사용 |
| 2 | Qwen3.5-35B-A3B | 20GB | AI-395 로컬 | MoE |
| 3 | GPT-4o | - | API | baseline (비용 참고용) |

**추후 확장 (RunPod):**
- Gemma4 계열
- Qwen3 최신
- Llama 4 계열
- 기타 대형 모델 (70B+)

**평가 방법:**
```
for llm in models:
    for (question, retrieved_chunks, target_answer) in eval_300:
        1. 프롬프트: system + top-K 청크 + question
        2. LLM 생성
        3. 자동 평가 (LLM-as-judge)
```

**자동 평가 (원본 재현):**
1. answer_similarity (threshold=4)
2. answer_correctness (threshold=4)
3. LLM-as-judge: GPT-4o 또는 Claude로 정답 대비 채점

**추가 실험:**
- Thinking ON vs OFF (Qwen3.5)
- 컨텍스트 수: top-3 vs top-5 vs top-10 청크
- 한국어 프롬프트 vs 영어 프롬프트

**예상:** 3 LLM × 300 = 900 생성, ~2~3시간

---

## 4. 전체 실험 매트릭스

| Phase | 변수 | 조합 수 | 질의 수 | 서버 |
|-------|------|---------|---------|------|
| 1 Parser | 3종 | 3 | 900 | 241 + 245 |
| 2 Chunking | 4종 | 4 | 1,200 | 241 + 245 |
| 3 VectorStore | 7종 | 7 | 2,100 | **250** + 245 |
| 4 Embedding | 28종 | 28 | 8,400 | 250 + **245** |
| 5 LLM | 3종 (+RunPod) | 3+ | 900+ | 245 / RunPod |
| **합계** | | **45+** | **~13,500** | |

---

## 5. 산출물

### 최종 비교표
```
| 순위 | Parser | Chunk | VectorStore | Embedding | LLM | Retrieval MRR | 정답률 |
|------|--------|-------|-------------|-----------|-----|--------------|--------|
| 1    | pymupdf4llm | 1500/200 | Qdrant | snowflake-ko | Qwen3.5-27B | 0.78 | ??.?% |
| ref  | PyPDF | 1000/200 | ? | OpenAI | GPT-4-Turbo | ? | 61.0% |
| ref  | (Alli 독점) | ? | ? | ? | Claude 3.5 | ? | 84.7% |
```

### 분석 리포트
- 컴포넌트별 기여도 (어디서 가장 큰 개선?)
- 한국어 특화 vs 다국어 임베딩 비교
- 모델 크기 vs 성능 efficiency frontier
- VectorStore 정확도 vs 속도 tradeoff
- 도메인별 최적 모델 차이
- context_type별 (paragraph/table/image) 취약점
- 상용 API vs 로컬 오픈소스 TCO

---

## 6. 참고 레포 (포크 완료)

| 레포 | ⭐ | 내용 | 포크 |
|------|---|------|------|
| Marker-Inc-Korea/AutoRAG-example-korean-embedding-benchmark | 44 | AutoRAG 임베딩 16종 | [BAEM1N/AutoRAG-example-korean-embedding-benchmark](https://github.com/BAEM1N/AutoRAG-example-korean-embedding-benchmark) |
| ssisOneTeam/Korean-Embedding-Model-Performance-Benchmark-for-Retriever | 50 | 임베딩 24종 | [BAEM1N/Korean-Embedding-Model-Performance-Benchmark-for-Retriever](https://github.com/BAEM1N/Korean-Embedding-Model-Performance-Benchmark-for-Retriever) |
| Songwooseok123/allganize-RAG-Evaluation-Dataset-KO-RAG- | 0 | allganize 실험 | [BAEM1N/allganize-RAG-Evaluation-Dataset-KO-RAG-](https://github.com/BAEM1N/allganize-RAG-Evaluation-Dataset-KO-RAG-) |

---

## 7. 일정

| 일차 | 작업 | 소요 |
|------|------|------|
| Day 1 | Phase 0: 데이터 준비 + T7910 벡터스토어 셋업 + 스크립트 | 반나절 |
| Day 1 | Phase 1: Parser 비교 | 30분 |
| Day 1 | Phase 2: Chunking 비교 | 30분 |
| Day 2 | Phase 3: VectorStore 비교 (T7910) | 1~2시간 |
| Day 2~3 | Phase 4: Embedding 비교 (28종) | 4~6시간 |
| Day 3 | Phase 5: LLM 비교 (로컬) | 2~3시간 |
| Day 4 | 분석 + 리포트 | 반나절 |
| 추후 | Phase 5 확장: RunPod + Gemma4/Qwen3/Llama4 | TBD |

**로컬 실험: 3~4일 → RunPod 확장: 추가 1~2일**

---

## 8. 리스크

| 리스크 | 대응 |
|--------|------|
| T7910 GPU 없음 → 벡터 검색 CPU only | 128GB RAM으로 HNSW 충분, GPU 불필요 |
| T7910 기존 서비스와 리소스 경합 | 벤치마크용 Docker 별도 네트워크, 메모리 제한 설정 |
| 임베딩 dim 차이 (512~4096) | 벡터스토어별 dim 동적 설정 |
| 대형 임베딩(8B) 로딩 시 LLM 불가 | Phase 4 그룹 C/D는 LLM 내리고 진행 |
| image context_type (57문항) | 텍스트 임베딩 한계, 별도 분석 |
| 자동 평가 오차 (~8%) | 5% 이내 차이는 동등 처리 |
