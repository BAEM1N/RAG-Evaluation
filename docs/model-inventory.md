# AI 모델 인벤토리

> 서버: 192.168.50.245 (AI-395)  
> GPU: AMD Instinct MI100 96GB VRAM  
> 모델 경로: `/home/baeumai/models/gguf/`  
> 총 디스크: 183GB / 468GB (58%)  
> 게이트웨이: `http://192.168.50.245:8080`  
> 최종 업데이트: 2026-04-09

---

## 실행 현황 (60 / 96 GB VRAM)

| 모델 | 용도 | VRAM | 포트 | 상시 |
|------|------|------|------|------|
| Qwen3-Embedding-8B | 임베딩 | 9 GB | auto | O |
| Qwen3-Reranker-4B | 리랭커 | 5 GB | auto | O |
| Qwen3.5-27B (Q4_K_M) | LLM | 26 GB | auto | O |
| Qwen3.5-35B-A3B (IQ4_XS) | LLM | 20 GB | auto | O |

여유 VRAM: **36 GB** (온디맨드 모델 로드 가능)

---

## LLM (채팅) — 3개

| # | alias | 파일 | 양자화 | 크기 | ctx | 상시 | 비고 |
|---|-------|------|--------|------|-----|------|------|
| 1 | qwen3.5-27b | Qwen3.5-27B-Q4_K_M.gguf | Q4_K_M | 16 GB | 262144 | O | Dense, 256K ctx |
| 2 | qwen3.5-35b-a3b | Qwen3.5-35B-A3B-UD-IQ4_XS.gguf | IQ4_XS | 17 GB | 262144 | O | MoE 3B active, 256K ctx |
| 3 | nemotron-120b | (별도 디렉토리) | IQ4_XS | ~44 GB | 65536 | X | MoE 12B active, 온디맨드 |

---

## 임베딩 — 28개

### Tier 1: 대형 (7B+)
| # | alias | 파일 | 크기 | dim | 특징 |
|---|-------|------|------|-----|------|
| 1 | qwen3-embed-8b | Qwen3-Embedding-8B-Q8_0.gguf | 7.5 GB | 4096 | **현재 사용**, MTEB 다국어 상위 |
| 2 | llama-embed-nemotron-8b | llama-embed-nemotron-8b.Q8_0.gguf | 7.5 GB | 4096 | MTEB 다국어 1위, decoder-based **(신규)** |
| 3 | nemotron-embed-8b | Nemotron-Embed-8B-Q8_0.gguf | 7.5 GB | 4096 | NVIDIA 구버전 |
| 4 | e5-mistral-7b-instruct | e5-mistral-7b-instruct-Q8_0.gguf | 7.2 GB | 4096 | MS, Mistral 기반 **(신규)** |

### Tier 2: 중형 (1B~4B)
| # | alias | 파일 | 크기 | dim | 특징 |
|---|-------|------|------|-----|------|
| 5 | qwen3-embed-4b | Qwen3-Embedding-4B-Q8_0.gguf | 4.0 GB | 4096 | |
| 6 | jina-v4-retrieval | jina-v4-retrieval-Q8_0.gguf | 3.1 GB | 4096 | 멀티모달 가능 |
| 7 | jina-v4-code | jina-v4-code-Q8_0.gguf | 3.1 GB | 4096 | 코드 특화 |
| 8 | jina-code-1.5b | jina-code-embeddings-1.5b-Q8_0.gguf | 1.6 GB | 1024 | 코드 전용 |

### Tier 3: 소형 (~335M, 한국어 특화)
| # | alias | 파일 | 크기 | dim | 특징 |
|---|-------|------|------|-----|------|
| 9 | snowflake-arctic-ko | snowflake-arctic-embed-l-v2.0-ko-Q8_0.gguf | 605 MB | 1024 | **한국어 1위** (84.77) **(신규, GGUF 자체변환)** |
| 10 | pixie-rune-v1 | PIXIE-Rune-v1.0-Q8_0.gguf | 605 MB | 1024 | **한국어 2위** (84.68) **(신규, GGUF 자체변환)** |
| 11 | kure-v1 | KURE-v1-Q8_0.gguf | 606 MB | 1024 | 한국어 4위 (83.10) |
| 12 | koe5 | koe5-q5_k_m.gguf | 417 MB | 1024 | 한국어 전용 E5 **(신규)** |

### Tier 4: 소형 (~335M, 다국어/범용)
| # | alias | 파일 | 크기 | dim | 특징 |
|---|-------|------|------|-----|------|
| 13 | qwen3-embed-0.6b | Qwen3-Embedding-0.6B-Q8_0.gguf | 610 MB | 1024 | |
| 14 | snowflake-arctic-l-v2 | snowflake-arctic-embed-l-v2.0-q8_0.gguf | 606 MB | 1024 | |
| 15 | bge-m3 | bge-m3-Q8_0.gguf | 606 MB | 1024 | 다국어, hybrid(dense+sparse+colbert) |
| 16 | me5-large-instruct | multilingual-e5-large-instruct-q8_0.gguf | 576 MB | 1024 | 다국어 |
| 17 | jina-v5-small-retrieval | v5-small-retrieval-Q8_0.gguf | 610 MB | 1024 | |
| 18 | harrier-0.6b | harrier-oss-v1-0.6b-Q8_0.gguf | 610 MB | 1024 | MS |
| 19 | labse | labse.Q8_0.gguf | 492 MB | 768 | Google, 109개 언어 |
| 20 | nomic-embed-v2-moe | nomic-embed-text-v2-moe.Q8_0.gguf | 489 MB | 768 | MoE |

### Tier 5: 초소형 (~100M)
| # | alias | 파일 | 크기 | dim | 특징 |
|---|-------|------|------|-----|------|
| 21 | mxbai-embed-large-v1 | mxbai-embed-large-v1.Q8_0.gguf | 342 MB | 1024 | BERT-large SOTA **(신규)** |
| 22 | voyage-4-nano | voyage-4-nano-q8_0.gguf | 355 MB | 1024 | MoE, Apache 2.0 **(신규)** |
| 23 | gemma-embed-300m | embeddinggemma-300M-Q8_0.gguf | 314 MB | 768 | Google |
| 24 | granite-278m | granite-embedding-278m-multilingual-Q8_0.gguf | 290 MB | 768 | IBM, 한국어 |
| 25 | harrier-270m | harrier-oss-v1-270M-Q8_0.gguf | 279 MB | 1024 | MS |
| 26 | jina-v5-nano-matching | v5-nano-text-matching-Q8_0.gguf | 223 MB | 512 | |
| 27 | granite-107m | granite-embedding-107m-multilingual-Q8_0.gguf | 116 MB | 768 | IBM |

### 특수: 초대형
| # | alias | 파일 | 크기 | dim | 특징 |
|---|-------|------|------|-----|------|
| 28 | harrier-27b | harrier-27b-Q8_0.gguf | 27 GB | 4096 | MS, 27B 임베딩 |

---

## 리랭커 — 9개

| # | alias | 파일 | 크기 | 언어 | 비고 |
|---|-------|------|------|------|------|
| 1 | qwen3-reranker-4b | Qwen3-Reranker-4B-Q8_0.gguf | 4.4 GB | 다국어 | **현재 사용** |
| 2 | qwen3-reranker-0.6b | Qwen3-Reranker-0.6B-Q8_0.gguf | 610 MB | 다국어 | |
| 3 | mxbai-rerank-large-v2 | mxbai-rerank-large-v2-q8_0.gguf | 1.6 GB | 영어 | |
| 4 | jina-reranker-m0 | jina-reranker-m0-Q8_0.gguf | 1.6 GB | 다국어 | |
| 5 | jina-reranker-v3 | jina-reranker-v3-Q8_0.gguf | 611 MB | 다국어 | |
| 6 | BGE-Reranker-v2-m3 | BGE-Reranker-v2-m3-Q8_0.gguf | 607 MB | 다국어 | |
| 7 | gte-multi-reranker | gte-multilingual-reranker-base-Q8_0.gguf | 317 MB | 다국어 | llama.cpp 미지원 (arch=new) |
| 8 | jina-reranker-v2-multi | jina-reranker-v2-base-multilingual-Q8_0.gguf | 292 MB | 다국어 | |
| 9 | bce-reranker | bce-reranker-base_v1-Q8_0.gguf | 290 MB | 중/영/일/**한** | 한국어 명시 지원 |

---

## HuggingFace 업로드 (자체 변환)

| 모델 | URL | 비고 |
|------|-----|------|
| snowflake-arctic-embed-l-v2.0-ko | [BAEM1N/snowflake-arctic-embed-l-v2.0-ko-GGUF](https://huggingface.co/BAEM1N/snowflake-arctic-embed-l-v2.0-ko-GGUF) | F16 + Q8_0 |
| PIXIE-Rune-v1.0 | [BAEM1N/PIXIE-Rune-v1.0-GGUF](https://huggingface.co/BAEM1N/PIXIE-Rune-v1.0-GGUF) | F16 + Q8_0 |

---

## 게이트웨이 미등록 (다운로드만 완료)

아래 모델들은 GGUF 다운로드 완료되었으나 `gateway/config.py`에 미등록. 필요시 추가 후 게이트웨이 재시작.

| 모델 | VRAM 예상 | 등록 alias 제안 |
|------|-----------|----------------|
| llama-embed-nemotron-8b | ~9 GB | `llama-embed-nemotron-8b` |
| e5-mistral-7b-instruct | ~8 GB | `e5-mistral-7b` |
| KoE5 | ~1 GB | `koe5` |
| voyage-4-nano | ~0.5 GB | `voyage-4-nano` |
| mxbai-embed-large-v1 | ~0.5 GB | `mxbai-embed-large` |
| gte-multilingual-reranker | ~0.5 GB | (llama.cpp 미지원) |

---

## 요약

| 카테고리 | 수량 | 비고 |
|----------|------|------|
| LLM (채팅) | 3 | 2 상시 + 1 온디맨드 |
| 임베딩 | 28 | 신규 7개 포함 |
| 리랭커 | 9 | |
| **합계** | **40** | 디스크 183GB |
