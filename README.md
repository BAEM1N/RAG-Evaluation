# RAG-Evaluation

한국어 RAG 파이프라인 컴포넌트별 벤치마크 실험.

**데이터셋:** [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)  
**범위:** 300 Q&A × 58 PDF × 5 도메인 (finance, public, medical, law, commerce)

---

## 목표

원본 allganize 실험은 상용 API(OpenAI, Cohere, Upstage)만 테스트.  
우리는 **100% 로컬 오픈소스**로 각 컴포넌트(Parser, Chunking, VectorStore, Embedding, LLM)를 분해 비교.

### 기존 연구 대비 차별점

| 항목 | allganize | AutoRAG | ssisOneTeam | **우리** |
|------|-----------|---------|-------------|---------|
| 데이터 | 300 Q&A | ? | 106 Q&A | 300 Q&A |
| Parser | PyPDF 고정 | ? | ? | **3종 비교** |
| Chunking | 1000/200 고정 | ? | ? | **4종 비교** |
| Embedding | OpenAI 1종 | 16종 혼합 | 24종 혼합 | **28종 (로컬 GGUF)** |
| VectorStore | 미명시 | ? | ? | **7종 비교** |
| LLM | GPT-4 (상용) | X | X | **로컬 + RunPod 확장** |

---

## 디렉토리 구조

```
RAG-Evaluation/
├── data/
│   ├── pdfs/                         # 58개 원본 PDF (도메인별)
│   │   ├── finance/     (10 PDFs)
│   │   ├── public/      (12 PDFs)
│   │   ├── medical/     (14 PDFs)
│   │   ├── law/         (12 PDFs)
│   │   └── commerce/    (10 PDFs)
│   ├── ground_truth.json             # 300 Q&A 원본
│   ├── ground_truth_filtered.json    # 매핑 가능한 Q&A
│   └── pdf_mapping.json              # {GT 파일명: 실제 파일명}
├── scripts/
│   ├── eval_utils.py                 # 공통 평가 유틸 (메트릭, PDF 파싱, 청킹)
│   └── bench_all.py                  # Phase 1~5 통합 실행 스크립트
├── results/
│   ├── phase1_parser/                # Parser 비교
│   ├── phase2_chunking/              # Chunking 비교
│   ├── phase3_vectorstore/           # VectorStore 비교
│   ├── phase4_embedding/             # Embedding 비교 (핵심)
│   └── phase5_llm/                   # LLM 생성 비교
├── docs/
│   ├── rag-benchmark-plan.md         # 전체 실험 계획서
│   └── model-inventory.md            # AI-395 모델 인벤토리
├── references/                       # 기존 연구 포크
│   ├── allganize-original/           # LangChain 파이프라인 노트북 (KoE5, Chroma, GPT-4o-mini, ko-reranker)
│   ├── AutoRAG-example-korean-embedding-benchmark/
│   └── Korean-Embedding-Model-Performance-Benchmark-for-Retriever/
└── docker-compose.vectorstores.yml   # T7910 벡터스토어 7종
```

---

## 인프라

| 서버 | IP | 역할 | 사양 |
|------|-----|------|------|
| AI-395 | 192.168.50.245 | 임베딩/LLM 모델 서빙 (게이트웨이) | MI100 96GB VRAM |
| Mac Mini | 192.168.50.241 | 실험 스크립트 | M2 16GB |
| **T7910** | **192.168.50.250** | **벡터스토어 서빙** | Dual Xeon 72T, 128GB RAM |

### T7910 벡터스토어 (7/7 가동)

| 스토어 | 포트 |
|--------|------|
| pgvector | 5433 |
| Chroma | 8100 |
| Milvus | 19530 |
| Weaviate | 8101 |
| Qdrant | 6340 |
| FAISS | (pip, 서버리스) |
| LanceDB | (pip, 서버리스) |

---

## 실험 Phase

| Phase | 변수 | 조합 | 질의 | 소요 |
|-------|------|------|------|------|
| 0 | 데이터 준비 | - | - | 완료 |
| 1 | Parser (PyPDF, pymupdf4llm, docling) | 3 | 900 | 30분 |
| 2 | Chunking (500~2000 × overlap) | 4 | 1,200 | 30분 |
| 3 | VectorStore (7종) | 7 | 2,100 | 1시간 |
| 4 | **Embedding (28종)** ⭐ | 28 | 8,400 | 4~6시간 |
| 5 | LLM (Qwen3.5-27B, 35B-A3B, +GPT-4o) | 3+ | 900+ | 2~3시간 |
| **합계** | | **45+** | **~13,500** | **~1.5일** |

자세한 계획: [docs/rag-benchmark-plan.md](docs/rag-benchmark-plan.md)

---

## 실행 방법

```bash
# Phase 1: Parser 비교
python scripts/bench_all.py --phase 1

# Phase 2: Chunking (Phase 1 최적 파서 사용)
python scripts/bench_all.py --phase 2 --parser pymupdf4llm

# Phase 4: Embedding 비교 (특정 모델만)
python scripts/bench_all.py --phase 4 --model snowflake-arctic-ko

# 전체 요약
python scripts/bench_all.py --summary
```

---

## 데이터 현황

| 항목 | 수량 |
|------|------|
| PDF | **58/58** (100%) |
| 질문 | **300/300** (100%) |
| finance | 60 Q&A |
| public | 60 Q&A |
| medical | 60 Q&A |
| law | 60 Q&A |
| commerce | 60 Q&A |
| **context_type** | paragraph 148, image 57, table 50, text 45 |

---

## 베이스라인: allganize LangChain Tutorial (공식)

**출처:** HuggingFace 데이터셋 페이지의 공식 [LangChain Tutorial Colab](https://colab.research.google.com/drive/1Jlzs8ZqFOqqIBBT2T5XGBhr23XxEsvHb)  
**노트북 사본:** `references/allganize_langchain_tutorial.ipynb`

### 파이프라인 구성

| 컴포넌트 | 구성 |
|---------|------|
| Parser | `PyPDFLoader` |
| Chunking | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)` |
| Embedding | `OpenAIEmbeddings()` (text-embedding-ada-002) |
| VectorStore | **Chroma** |
| Retriever | similarity, k=6 |
| LLM | **gpt-4-turbo** |
| Prompt | LangChain Hub `rlm/rag-prompt` |
| Output | `StrOutputParser` |

**리더보드 결과:** GPT-4-Turbo 기준 **61.0%** (183/300)

### 우리 실험의 변수 치환 방식

각 Phase에서 이 baseline의 **한 컴포넌트만** 바꿔가며 비교:

```
[baseline]    PyPDFLoader → RecursiveSplit(1000/200) → OpenAIEmbedding → Chroma → k=6 → rlm/rag-prompt → GPT-4-Turbo

[Phase 1]     {PyPDF, pymupdf4llm, docling}  → (나머지 고정)
[Phase 2]     (Phase1 최적) → {500, 1000, 1500, 2000} × overlap
[Phase 3]     (Phase1~2 고정) → {pgvector, FAISS, Chroma, Milvus, Qdrant, Weaviate, LanceDB}
[Phase 4] ⭐  (Phase1~3 고정) → {qwen3-embed-8b, llama-embed-nemotron, ... 28종}
[Phase 5]     (Phase1~4 고정) → {Qwen3.5-27B, Qwen3.5-35B-A3B, GPT-4o, ...}
```

**Retrieval 메트릭**: Hit@k, MRR, NDCG (Phase 1~4)  
**Generation 메트릭**: allganize 자동평가(O/X) 재현 (Phase 5)

---

## 기타 참고 레포

| 레포 | 내용 | 위치 |
|------|------|------|
| Songwooseok123/allganize-RAG | 개인 LangChain 구현 (KoE5+Chroma+ko-reranker+GPT-4o-mini) | `references/allganize-original/` |
| Marker-Inc/AutoRAG-example | AutoRAG 기반 한국어 임베딩 16종 | `references/AutoRAG-example-korean-embedding-benchmark/` |
| ssisOneTeam/Korean-Embedding | 임베딩 24종 복지 도메인 | `references/Korean-Embedding-Model-Performance-Benchmark-for-Retriever/` |

---

## 사용 모델

### 임베딩 (AI-395, 28종)
| Tier | 모델 |
|------|------|
| Large (7B+) | qwen3-embed-8b, llama-embed-nemotron-8b, nemotron-embed-8b, e5-mistral-7b |
| Medium (1~4B) | qwen3-embed-4b, jina-v4-retrieval, jina-v4-code, jina-code-1.5b |
| Korean (~335M) | **snowflake-arctic-ko**, **pixie-rune-v1**, kure-v1, koe5 |
| Multilingual (~335M) | qwen3-0.6b, snowflake-l-v2, bge-m3, me5-large, jina-v5-small, harrier-0.6b, labse, nomic-v2-moe |
| Tiny (<300M) | mxbai-embed-large, voyage-4-nano, gemma-300m, granite-278m, harrier-270m, jina-v5-nano, granite-107m |
| XL | harrier-27b |

**자체 변환 GGUF:**
- [BAEM1N/snowflake-arctic-embed-l-v2.0-ko-GGUF](https://huggingface.co/BAEM1N/snowflake-arctic-embed-l-v2.0-ko-GGUF) — 한국어 1위 (84.77)
- [BAEM1N/PIXIE-Rune-v1.0-GGUF](https://huggingface.co/BAEM1N/PIXIE-Rune-v1.0-GGUF) — 한국어 2위 (84.68)

### LLM (AI-395)
- Qwen3.5-27B (26GB)
- Qwen3.5-35B-A3B (20GB, MoE)
- (추후) RunPod + Gemma4/Llama4 확장
