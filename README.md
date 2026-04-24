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
| Embedding | OpenAI 1종 | 16종 혼합 | 24종 혼합 | **27종 (로컬 GGUF)** |
| VectorStore | 미명시 | ? | ? | **7종 비교** |
| LLM 생성 | GPT-4 (상용) | X | X | **12종 (AI-395 llama.cpp + DGX Spark ollama)** |
| Judge | — | — | — | **6종 LLM-as-Judge (allganize 방식, 4 metric × majority vote)** |

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

| 서버 | 역할 | 사양 |
|------|------|------|
| AI-395 | 임베딩/LLM 모델 서빙 (llama.cpp) | 96GB VRAM |
| DGX Spark | 대형 LLM 판정 (ollama) | 단일 GB10, 통합 메모리 128GB |
| Mac Mini | 실험 스크립트 실행 | M4 16GB |
| **T7910** | **벡터스토어 서빙** | Dual Xeon 72T, 128GB RAM |

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
| 1 | Parser (pypdf, pymupdf, pymupdf4llm) | 3 | 900 | 30분 |
| 2 | Chunking (500~2000 × overlap) | 4 | 1,200 | 30분 |
| 3 | VectorStore (7종) | 7 | 2,100 | 1시간 |
| 4 | **Embedding (27종)** ⭐ | 27 | 8,100 | 4~6시간 |
| 5A | LLM 생성 (12종) | 12 | 3,600 | 1~2일 |
| 5B | LLM-as-Judge (6 judge × 12 LLM × 4 metric) | 72 | 86,400 | 3~5일 |
| **합계** | | **105+** | **~102,000** | **~5-7일** |

**현재 진행 상태** (2026-04-24):
- ✅ Phase 1~4 완료 — [results/phase4_embedding/LEADERBOARD.md](results/phase4_embedding/LEADERBOARD.md)
- 🔄 Phase 5 생성 12/12 완료, 판정 진행 중 (AI-395 claude-distill + DGX Spark qwen3-next)
- ⏳ solar-open-100b judge 재판정 대기 (ollama custom modelfile의 빈 chat template → llama-server `--jinja`로 GGUF 내장 Jinja 사용 예정)

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

### 임베딩 (AI-395, 27종 실측 완료)

Phase 4 최종 리더보드 (MRR 상위 10):

| Rank | Model | dim | MRR | Hit@1 | Hit@5 |
|---:|---|---:|---:|---:|---:|
| 🥇 | **koe5** | 1024 | 0.6871 | 60.7% | 80.7% |
| 🥈 | gemma-embed-300m | 768 | 0.6650 | 57.3% | 79.7% |
| 🥉 | pixie-rune-v1 | 1024 | 0.6627 | 58.7% | 76.0% |
| 4 | snowflake-arctic-ko | 1024 | 0.6612 | 58.3% | 75.0% |
| 5 | snowflake-arctic-l-v2 | 1024 | 0.6495 | 58.3% | 73.0% |

전체 27종 결과: [results/phase4_embedding/LEADERBOARD.md](results/phase4_embedding/LEADERBOARD.md)

**자체 변환 GGUF:**
- [BAEM1N/snowflake-arctic-embed-l-v2.0-ko-GGUF](https://huggingface.co/BAEM1N/snowflake-arctic-embed-l-v2.0-ko-GGUF)
- [BAEM1N/PIXIE-Rune-v1.0-GGUF](https://huggingface.co/BAEM1N/PIXIE-Rune-v1.0-GGUF)

### Phase 5 생성 LLM (12종)

AI-395(llama.cpp) + DGX Spark(ollama) 혼합 운영:

- **Qwen3.5 시리즈**: 9b-Q4/Q8, 27b-Q8, 122b-a10b-Q4 (think/nothink 각 1)
- **DeepSeek-R1**: 70B (nothink)
- **GPT-OSS**: 20B, 120B (AI-395)
- **ExaOne**: 3.5-32B
- **Mistral-Small**: 24B
- **Phi**: 4-14B
- **LFM**: 2-24B

### Phase 5 Judge (LLM-as-Judge, 6종 + 1 제외)

allganize 원본 방식: MLflow answer_similarity/v1 + correctness/v1 기반 4 metric × threshold 4 × majority vote → O/X 판정

| Judge | 장비 | 상태 |
|---|---|---|
| gemma4:31b | Mac ollama | ✅ 12/12 |
| nemotron-120b | AI-395 | ✅ 12/12 |
| qwen3.5:122b-a10b | Mac ollama | ✅ 12/12 |
| qwen3.6-35b-a3b | AI-395 | ✅ 12/12 |
| supergemma4-26b | AI-395 | ✅ 12/12 |
| qwen3-next:80b | DGX Spark | 🔄 진행 중 |
| qwen3.5-27b-claude-distill | AI-395 | 🔄 진행 중 |
| solar-open-100b | DGX Spark | ⏳ 재판정 대기 (llama-server `--jinja`로 재기동 예정) |

전체 판정 매트릭스 & 중간 리더보드: [results/phase5_judge/LEADERBOARD.md](results/phase5_judge/LEADERBOARD.md)
