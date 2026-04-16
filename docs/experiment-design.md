# 실험 설계 (Experiment Design)

> 한국어 RAG 파이프라인 컴포넌트별 벤치마크 — 실험 구조, 변수, 측정 방법 정의

## 1. 연구 질문

본 프로젝트는 한국어 RAG에서 **각 컴포넌트가 최종 성능에 얼마나 기여하는가**를 정량화한다.

- RQ1. Parser가 RAG 성능에 미치는 영향은?
- RQ2. Chunk 크기/오버랩이 검색 정확도에 미치는 영향은?
- RQ3. VectorStore 7종은 실질적 성능 차이가 있는가?
- RQ4. 임베딩 모델별 한국어 검색 성능은 어떻게 다른가?
- RQ5. LLM 선택(로컬 양자화 vs 상용 API)이 답변 품질에 미치는 영향은?
- RQ6. Thinking(reasoning) 모드가 답변 품질 향상에 기여하는가?

## 2. 데이터셋

[allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)

| 항목 | 수량 |
|------|------|
| 질문 | 300 |
| 도메인 | finance / public / medical / law / commerce (각 60) |
| PDF | 58 |
| Context type | paragraph 148 / image 57 / table 50 / text 45 |

각 질문에는 정답 PDF 파일명, 정답 페이지, 정답 텍스트가 주어진다.

## 3. 실험 구조 (Phased)

각 Phase에서 **다른 컴포넌트는 고정하고 하나만 변경**한다. 이전 Phase 1위를 다음 Phase의 고정값으로 사용.

```
[baseline] PyPDFLoader → RecursiveSplit(1000/200) → OpenAIEmbedding → Chroma → gpt-4-turbo

Phase 1: {pypdf, pymupdf4llm, pymupdf}              → 나머지 고정
Phase 2: (Phase 1 winner) → {500, 1000, 1500, 2000} × overlap
Phase 3: (Phase 1~2 winner) → {pgvector, FAISS, Chroma, Milvus, Qdrant, Weaviate, LanceDB}
Phase 4: (Phase 1~3 winner) → {21 embedding models}
Phase 5: (Phase 1~4 winner) → {~30 LLMs, local + OpenRouter + Friendli.ai}
```

### 실험 규모

| Phase | 변수 | 조합 수 | 질의 수 |
|-------|------|---------|---------|
| 1 Parser | 3 | 3 | 900 |
| 2 Chunking | 4 | 4 | 1,200 |
| 3 VectorStore | 7 | 7 | 2,100 |
| 4 Embedding | 21 | 21 | 6,300 |
| 5 LLM | ~30 | ~30 | ~9,000 |
| **합계** | | **~65** | **~19,500** |

## 4. 고정 변수 (Baseline Frozen Values)

모든 Phase는 이 baseline의 **한 컴포넌트만** 바꾼다.

| 컴포넌트 | 값 | 결정 근거 |
|---------|------|----------|
| Parser | pymupdf4llm | Phase 1 MRR 1위 (0.4715) |
| Chunking | 500 / overlap 100 | Phase 2 MRR 1위 (0.5315) |
| VectorStore | FAISS | Phase 3 속도 1위 (정확도 동률) |
| Embedding (Phase 5용) | gemma-embed-300m | Phase 4 MRR 1위 (0.6682) |
| Top-k | 5 | allganize 원본 k=6과 유사 |
| 트렁케이트 | 500자 | llama.cpp 512 토큰 제한 대응 |

## 5. 측정 지표

### Retrieval (Phase 1~4)

| 지표 | 정의 | 용도 |
|------|------|------|
| Hit@1 | top-1이 정답 페이지인 비율 | 최상위 검색 정확도 |
| Hit@5 | top-5 내 정답 페이지 존재 비율 | 실용 검색 정확도 |
| File Hit@5 | top-5 내 정답 파일 존재 비율 | 파일 수준 정확도 |
| **MRR** | 평균 역순위 | 주요 지표 (순위 반영) |
| NDCG@5 | 순위 가중 누적 이득 | 보조 지표 |

### Generation (Phase 5)

| 지표 | 정의 | 도구 |
|------|------|------|
| Answer correctness | 정답 대비 사실 일치도 (1~5점) | LLM-as-judge (gpt-5.4) |
| Answer similarity | 의미 유사도 (1~5점) | LLM-as-judge |
| Faithfulness | 환각 없음 (1~5점) | LLM-as-judge |
| Completeness | 정답 요점 커버리지 (1~5점) | LLM-as-judge |
| Context Precision | top-k 중 관련 chunk 비율 | RAGAS |
| Context Recall | 정답을 뒷받침하는 정보의 context 커버리지 | RAGAS |

## 6. 실험 A / B 구조 (Phase 5)

**실험 A: 임베딩 영향 측정**
- 21 임베딩 × 4 고정 LLM (qwen3.5-27b/35b-a3b × think/nothink) × 300 질문
- 조합: 84개 → 결과: `results/phase5_exp_a_embed/`

**실험 B: LLM 영향 측정**
- 1 고정 임베딩 (gemma-embed-300m) × ~30 LLM × 300 질문
- 조합: 30개 → 결과: `results/phase5_exp_b_provider/`

## 7. LLM 분담

| 위치 | 모델 | 하드웨어 | 계정 |
|------|------|---------|------|
| AI-395 (llama.cpp) | qwen3.5-27b, 35b-a3b, 9b, EXAONE 4.5, Midm 2.0, SuperGemma4-26b, Qwen3.5-27b Claude Distilled | MI100 96GB VRAM | - |
| DGX Spark (Ollama) | qwen3.5 계열 (27b/35b/122b), gpt-oss 20b/120b, deepseek-r1, gemma4, llama4-scout, exaone3.5/4.0, phi4, mistral-small, lfm2 | GB10 128GB unified | - |
| OpenRouter | GPT-5.4, Claude 4.6, Gemini 3.1, Grok 4.20, Qwen3.5/3.6, GLM-5.1, Kimi K2.5, DeepSeek V3.1, Mistral Small, Cohere Command A, Perplexity Sonar | - | OPENROUTER_API_KEY |
| Friendli.ai | K-EXAONE 236B, Qwen3-235B, DeepSeek V3.2, Llama 3.3-70B, Llama 3.1-8B | - | FRIENDLI_TOKEN |

## 8. 병렬화

### 서버측

- llama.cpp: `-np 8` (8 슬롯)
- Ollama: `OLLAMA_NUM_PARALLEL=8`

### 클라이언트측

- AI-395 scripts: `ThreadPoolExecutor(max_workers=4)`
- Spark scripts: `ThreadPoolExecutor(max_workers=8)`
- OpenRouter batch: `llm.batch(prompts, config={"max_concurrency": 20})`

## 9. 비용/시간

### 시간 (실측 기반)

| Phase | 소요 |
|-------|------|
| 1 Parser | 1시간 |
| 2 Chunking | 1시간 |
| 3 VectorStore | 10분 (임베딩 1회 재사용) |
| 4 Embedding (21종) | 6시간 (대부분 로컬) |
| 5 LLM 실험 A (84조합) | 3~4일 (AI-395 + Spark 병렬) |
| 5 LLM 실험 B (30조합) | 1시간 (API 배치) |

### 비용

- Phase 1~4, 실험 A: **$0** (100% 로컬)
- 실험 B (OpenRouter + Friendli.ai): **~$47** (dry-run 추정)
- RAGAS 평가 (gpt-5.4-mini): **~$20**
- **총 $70 이내**

## 10. 재현 방법

```bash
# 환경 설정
uv sync

# Phase 1~4 (로컬)
python scripts/bench_all.py --phase 1
python scripts/bench_all.py --phase 2 --parser pymupdf4llm
python scripts/bench_all.py --phase 3 --parser pymupdf4llm --chunk-size 500 --chunk-overlap 100
python scripts/bench_phase4_parallel.py

# 캐시 사전 계산 (실험 A/B 공통)
python scripts/precompute_retrieval.py

# Phase 5 실험 A (AI-395 + Spark)
python scripts/phase5_experiments.py --exp A --server ai395
python scripts/phase5_experiments.py --exp A --server spark

# Phase 5 실험 B (API)
export OPENROUTER_API_KEY="sk-or-v1-..."
export FRIENDLI_TOKEN="flp_..."
python scripts/phase5_provider.py --provider all

# 분석
python scripts/analyze_retrieval_deep.py --csv
python scripts/evaluate_ragas.py --judge gpt-5.4-mini
```

## 11. 알려진 이슈와 대응

| 이슈 | 원인 | 대응 |
|------|------|------|
| 빈 임베딩 응답 | llama.cpp 512 토큰 제한 | 500자 트렁케이트 |
| Qwen3.5 thinking 토큰 낭비 | 기본값 thinking ON | `chat_template_kwargs: {enable_thinking: False}` |
| pgvector 인덱스 실패 | 4096차원 > HNSW 2000 제한 | dim > 2000 시 순차 검색 |
| Ollama 모델 확인 지연 | 첫 로드 시간 30초+ | `ollama list`로 선확인 |
| harrier-27b 랜덤 출력 | GGUF 양자화 문제 | 결과에서 제외 |

## 12. 공개 산출물

- `data/ground_truth.json` — 매핑 완료된 300 Q&A
- `data/prepared_chunks.json` — 최적 파서/청킹 청크
- `data/retrieval_cache/` — 21 임베딩 × 300 질문 검색 결과 (31MB)
- `results/phase1~5_*/` — 각 Phase JSON 결과
- `results/retrieval_analysis/` — 캐시 기반 심층 분석 CSV
- `scripts/*.py` — 모든 벤치마크 코드
- `docs/*.md` — 실험 설계 / 결과 / 모델 인벤토리
