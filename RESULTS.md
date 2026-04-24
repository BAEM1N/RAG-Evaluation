# RAG-Evaluation 실험 결과 스냅샷

**업데이트**: 2026-04-24  
**데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDF)

각 Phase는 **이전 Phase 최적값 고정 + 해당 변수만 치환** 방식으로 단일 변수 비교.

---

## Phase 1 — Parser (3종)

| Parser | MRR | Hit@1 | Hit@5 | File@5 | 청크 수 | 파싱 방식 |
|--------|----:|------:|------:|-------:|--------:|----------|
| **pymupdf4llm** 🥇 | **0.4715** | 38.3% | 58.3% | 86.0% | 1,920 | 마크다운 변환 (헤더·테이블) |
| pymupdf | 0.4663 | 35.7% | 63.3% | 86.3% | 1,263 | 일반 텍스트 |
| pypdf | 0.4472 | 34.3% | 60.7% | 82.7% | 1,224 | 라인 기반 |

**결정**: pymupdf4llm 고정 (MRR +5.4%).

---

## Phase 2 — Chunking (4종)

고정: Parser=pymupdf4llm

| 전략 | chunk_size / overlap | MRR | Hit@1 | Hit@5 | 청크 수 |
|------|---------------------:|----:|------:|------:|--------:|
| **small** 🥇 | 500 / 100 | **0.5315** | 45.0% | 65.0% | 3,166 |
| baseline 🥈 | 1000 / 200 | 0.4713 | 38.3% | 58.3% | 1,920 |
| medium | 1500 / 200 | 0.4458 | 36.3% | 55.0% | 1,468 |
| large | 2000 / 300 | 0.4302 | 34.3% | 53.3% | 1,370 |

**결정**: 500/100 고정 (MRR **+23.5%** — 모든 Phase 중 최대 영향).

---

## Phase 3 — VectorStore (7종)

고정: Parser=pymupdf4llm, Chunking=500/100, Embedding=qwen3-embed-8b

| 스토어 | MRR | Hit@5 | Insert(s) | p95 latency(ms) | QPS |
|--------|----:|------:|----------:|----------------:|----:|
| **FAISS** 🥇 | 0.5304 | 65.0% | **0.76** | **0.74** | **1394** |
| LanceDB | 0.5304 | 65.0% | 6.04 | 6.61 | 158 |
| Qdrant | 0.5304 | 65.0% | 58.58 | 122.20 | 9 |
| Milvus | 0.5304 | 65.0% | 22.39 | 57.47 | 19 |
| Weaviate | 0.5298 | 64.7% | 12.00 | 26.86 | 43 |
| Chroma | 0.5271 | 64.7% | 16.72 | 46.34 | 25 |
| pgvector | 0.5304 | 65.0% | 92.32 | 174.23 | 7 |

**결정**: FAISS 고정 (정확도 동률 + 속도 200배).

---

## Phase 4 — Embedding (27종) ⭐

고정: Parser=pymupdf4llm, Chunking=500/100, VectorStore=FAISS

상위 10개 + 주요 실패:

| Rank | 모델 | dim | MRR | Hit@1 | Hit@5 | 비고 |
|---:|---|---:|---:|---:|---:|---|
| 🥇 | **koe5** | 1024 | 0.6871 | 60.7% | 80.7% | 한국어 파인튜닝 600M |
| 🥈 | gemma-embed-300m | 768 | 0.6650 | 57.3% | 79.7% | Best tiny |
| 🥉 | pixie-rune-v1 | 1024 | 0.6627 | 58.7% | 76.0% | |
| 4 | snowflake-arctic-ko | 1024 | 0.6612 | 58.3% | 75.0% | 한국어 튜닝 |
| 5 | snowflake-arctic-l-v2 | 1024 | 0.6495 | 58.3% | 73.0% | |
| 6 | jina-v4-retrieval | 4096 | 0.6449 | 54.7% | 78.7% | |
| 7 | nomic-embed-v2-moe | 768 | 0.6435 | 56.7% | 75.3% | MoE |
| 8 | kure-v1 | 1024 | 0.6267 | 54.7% | 74.3% | 한국어 |
| 9 | harrier-0.6b | 1024 | 0.6131 | 53.3% | 70.3% | pooling=last |
| 10 | granite-278m | 768 | 0.5969 | 50.3% | 72.0% | |
| … | (17~25위) | | 0.5~0.1 | | | |
| 25 | mxbai-embed-large | 1024 | 0.1157 | 8.7% | 15.7% | 영어 전용 |
| 26 | labse | 768 | 0.0472 | 2.7% | 8.0% | 구버전 |
| 27 | harrier-27b | 5376 | 0.0170 | 1.0% | 2.3% | 한국어 부적합 |

**결정**: koe5 (생성/판정 공통) — 단 판정 실험에는 gemma-embed-300m 사용(2위, 가볍고 빠름).

**주요 발견**:
- 한국어 소형 모델이 대형 영어 모델을 압도 (koe5 600M > qwen3-embed-8b 0.16 MRR 차)
- 한국어 특화 4개(koe5, snowflake-ko, kure, pixie)가 top 8에 포함
- harrier-27b는 variance < 0.0001로 97% dim 죽음 (한국어 쿼리-문서 분리 실패)
- nemotron-embed-8b ≡ llama-embed-nemotron-8b (동일 아키텍처/가중치)

전체: [results/phase4_embedding/LEADERBOARD.md](results/phase4_embedding/LEADERBOARD.md)

---

## Phase 5A — 생성 LLM (12종)

고정: Parser/Chunking/VectorStore + Embedding=gemma-embed-300m (판정 친화적), top-5

| LLM | 장비 | 양자화 | think |
|---|---|---|---|
| deepseek-r1:70b | Spark | Q4 | nothink |
| exaone3.5:32b | Spark | Q4 | — |
| gpt-oss:120b | AI-395 | MXFP4 | — |
| gpt-oss:20b | AI-395 | MXFP4 | — |
| lfm2:24b | Spark | Q4 | — |
| mistral-small:24b | Spark | Q4 | — |
| phi4:14b | Spark | Q4 | — |
| qwen3.5:9b | Mixed | Q4_K_M / Q8_0 | nothink (2개 config) |
| qwen3.5:27b | AI-395 | Q8_0 | nothink |
| qwen3.5:122b-a10b | Spark | Q4_K_M | nothink |
| qwen3.5:122b-a10b | Spark | Q4_K_M | **think** |

---

## Phase 5B — LLM-as-Judge (6/7 진행)

**판정 방법** (allganize 원본 재현):
1. 4 metric 평가 — similarity, correctness, completeness, faithfulness (MLflow v1 rubric, 1–5 scale)
2. threshold=4 → metric별 O/X
3. Majority vote (≥2 metric O → 해당 Q는 O)
4. Accuracy = O 개수 / 총 질문 수

### 현재 판정 매트릭스 (6 judge × 12 LLM = 72 cell)

| Judge | 장비 | 완료 |
|---|---|---:|
| gemma4:31b (nothink) | Mac ollama | ✅ 12/12 |
| nemotron-120b (nothink) | AI-395 llama-server | ✅ 12/12 |
| qwen3.5:122b-a10b (nothink) | Mac ollama | ✅ 12/12 |
| qwen3.6-35b-a3b (nothink) | AI-395 llama-server | ✅ 12/12 |
| supergemma4-26b (nothink) | AI-395 llama-server | ✅ 12/12 |
| **qwen3-next:80b** (nothink) | DGX Spark ollama | 🔄 3/12 진행 중 |
| **qwen3.5-27b-claude-distill** (nothink) | AI-395 llama-server | 🔄 0/12 진행 중 |
| **solar-open-100b** | DGX Spark → 재설정 | ⏳ 재판정 대기 |

**solar-open-100b 1차 실패 & 재판정 계획**:
- 1차 결과: 1,200 vote 전부 0점 (파싱 실패 100%), 12 파일 모두 acc=0.000
- 원인: ollama custom modelfile의 `TEMPLATE {{ .Prompt }}` 가 chat template 생략 → glm4moe 모델이 completion 모드로 동작 → 1-5 정수 출력 안 나옴
- 재판정: ollama 우회, **llama-server `--jinja`** 옵션으로 GGUF 내장 공식 Jinja template 직접 사용
  - Solar-Open-100B의 chat 포맷: `<|begin|>role<|content|>msg<|end|>` + 별도 `<|think|>` 블록
  - 반영 시 qwen3-next / claude-distill 처럼 정상 1-5 출력 기대
- 실행 시점: qwen3-next:80b judge 12/12 완료 후 DGX Spark에서 이어 진행

### 중간 리더보드 (5-judge 기준, partial)

| Rank | LLM | 평균 acc | Judge 일치도 범위 |
|---:|---|---:|---|
| 1 | gpt-oss:120b | 0.77 | 0.73–0.80 |
| 2 | gpt-oss:20b | 0.76 | 0.71–0.81 |
| 3 | qwen3.5:122b-a10b (think) | 0.72 | 0.68–0.75 |
| 4 | qwen3.5:27b-q8 (nothink) | 0.72 | 0.67–0.74 |
| 5 | qwen3.5:122b-a10b (nothink) | 0.71 | 0.67–0.74 |
| 6 | exaone3.5:32b | 0.70 | 0.62–0.72 |
| 7 | phi4:14b | 0.70 | 0.62–0.73 |
| 8 | mistral-small:24b | 0.67 | 0.62–0.72 |
| 9 | qwen3.5:9b-q8 | 0.63 | 0.56–0.67 |
| 10 | qwen3.5:9b-q4 | 0.64 | 0.58–0.68 |
| 11 | deepseek-r1:70b | 0.62 | 0.36–0.72 (claude-distill 0.36 편향) |
| 12 | lfm2:24b | 0.52 | 0.39–0.57 |

완전 집계는 qwen3-next + claude-distill 판정 완주 후 확정.

---

## 한국어 RAG 최적 조합 (실험 결과 기반)

```
Parser:       pymupdf4llm
Chunking:     chunk_size=500, overlap=100
VectorStore:  FAISS
Embedding:    koe5 (또는 gemma-embed-300m)
top-k:        5
LLM:          gpt-oss:120b (Phase 5B 1위 예상) 또는 Qwen3.5:27b Q8 (로컬 구동 가능 + 상위권)
```

---

## 재현

```bash
# Phase 별 실행 (이전 Phase 최적값 자동 반영)
python scripts/bench_all.py --phase 1
python scripts/bench_all.py --phase 4 --model koe5
python scripts/phase5_batch_generate.py
python scripts/llm_judge.py <expB__*.json>
python scripts/judge_leaderboard.py  # 최종 집계
```

실험 설계: [docs/experiment-design.md](docs/experiment-design.md)  
비용 산출 (프로바이더 API 확장 계획): [docs/cost-report.md](docs/cost-report.md)  
데모 웹 서비스 계획: [docs/rag-demo-plan.md](docs/rag-demo-plan.md)
