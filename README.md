# Korean RAG Evaluation Benchmark

> 한국어 RAG 파이프라인 6개 stage를 단변량과 Cartesian 조합으로 분해해, **무엇이 실제 답변 품질을 올리는지** 검증한 공개 벤치마크입니다.

[![dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue)](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark)
[![repo](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/BAEM1N/RAG-Evaluation)

이 저장소는 **소스코드와 단계별 분석 보고서**를 담습니다. 원문 코퍼스와 실험 결과 데이터는 HuggingFace에 공개되어 있으며, GitHub에는 포함하지 않습니다.

---

## 결론

**RAG 파이프라인 최적화가 모델 업그레이드보다 컸습니다.** 동일한 GPT-5.4를 쓰더라도 retrieval/reranker/PreR 조합을 최적화하면 GPT-5.4-pro보다 **+6.0pp 높은 accuracy**를 기록했습니다.

| Pipeline | Generator | Accuracy |
|---|---|---:|
| **Cartesian winner**: query2doc + Hybrid 7:3 + jina-reranker-m0 | GPT-5.4 | **0.827** |
| Phase 5 단순 retrieval + GPT-5.4 | GPT-5.4 | 0.787 |
| Phase 5 + GPT-5.4-pro | GPT-5.4-pro | 0.767 |
| Phase 5 오픈 가중치(open weights) 1위 | gpt-oss_120b / kimi-k2.5 | 0.740 |

- 동일 GPT-5.4에서 파이프라인 최적화만으로 **+4.0pp accuracy** (0.787 → 0.827)
- GPT-5.4-pro로 업그레이드한 경우보다 **+6.0pp accuracy**
- 비용을 더 큰 클로즈 가중치(closed weights) 모델에만 쓰기보다, retrieval/reranker/PreR 조합 탐색에 쓰는 편이 효과적이었습니다.

## 데이터 / 규모

| 항목 | 내용 |
|---|---|
| 원본 코퍼스 | [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO) |
| 공개 결과 데이터 | [BAEM1N/Korean-RAG-LLM-Judge-Benchmark](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark) |
| 규모 | 300 Q&A × 58 PDFs × 5 도메인 |
| 단변량 평가 | 6 stage, 576 configs 측정값 |
| Cartesian 평가 | 384 configs = 8 PreR × 6 Retriever × 8 PostR |
| Stage 6 호출량 | 576,000 GPT-5.4 호출 (generation + judge) |
| 평가 규칙 | 18-judge 4-metric majority O: similarity / correctness / completeness / faithfulness |

## Winner Pipeline

| 목적 | Pipeline | Metric |
|---|---|---:|
| Accuracy / Judge 최고 | query2doc + Hybrid 7:3 + jina-reranker-m0 | **acc 0.827 / judge 4.067** |
| MRR / Hit@1 최고 | multi_query_para + Hybrid 5:5 + jina-reranker-m0 | **MRR 0.7874 / Hit@1 75.0%** |

```text
PyMuPDFLoader
  → RecursiveCharacterTextSplitter(300, 50)
  → google/embeddinggemma-300m
  → query2doc (PreR, GPT-5.4)
  → Hybrid 7:3 (Dense + BM25-KIWI)
  → top-20
  → jinaai/jina-reranker-m0
  → top-5
  → GPT-5.4 답변
```

## 누적 개선

naive baseline은 dense only + no rerank 조합입니다.

| 단계 | MRR | Hit@1 | Judge | Accuracy |
|---|---:|---:|---:|---:|
| Baseline | 0.6816 | 59.0% | 3.850 | — |
| + Hybrid 검색 | 0.7171 | 65.3% | 3.869 | — |
| + Reranker | 0.7697 | 74.0% | 3.916 | — |
| **Cartesian Judge / Accuracy 최고** | 0.7630 | 71.3% | **4.067** | **0.827** |
| **MRR 최고 조합** | **0.7874** | **75.0%** | 3.991 | 0.790 |

최대 누적 개선은 **MRR +15.5%**, **Hit@1 +27.1% 상대 개선(+16.0pp)**, **Judge +5.6%**입니다. Hit@1 기준으로는 300문제 중 48문제를 추가로 맞춘 셈입니다.

## 단계별 보고서

| Stage | 문서 | Winner / 핵심 결과 | Metric |
|---|---|---|---:|
| 1. Loader | [`docs/1_loader.md`](docs/1_loader.md) | PyMuPDFLoader | MRR 0.6486 |
| 2. Parser | [`docs/2_parser.md`](docs/2_parser.md) | LC Recursive 300/50 | MRR 0.6816 (dense) / 0.7171 (hybrid) |
| 3. Embedding | [`docs/3_embedding.md`](docs/3_embedding.md) | KoE5 / embeddinggemma-300m | MRR 0.6871 |
| 4. Retriever | [`docs/4_retriever.md`](docs/4_retriever.md) | Hybrid 3:7 (Dense + BM25-KIWI) | MRR 0.7171 |
| 4-1. Pre-Retriever | [`docs/4-1_pre_retriever.md`](docs/4-1_pre_retriever.md) | query_expansion (e2e judge 1위) | Judge 3.998 / MRR 0.7783 |
| 4-2. Post-Retriever | [`docs/4-2_post_retriever.md`](docs/4-2_post_retriever.md) | dragonkue/bge-reranker-v2-m3-ko (+1.83pp vs Qwen3-Reranker-4B) | MRR 0.7697 |
| 5. Generation | [`docs/5_generation.md`](docs/5_generation.md) | 오픈 가중치 vs 클로즈 가중치 leaderboard | 46 gen × 18 judge |
| 5-ext. e2e Axis-wise | [`docs/5_generation_e2e_axis.md`](docs/5_generation_e2e_axis.md) | Hybrid 5:5 + ko-reranker | Judge 3.983 |
| 6. Cartesian | [`docs/6_cartesian.md`](docs/6_cartesian.md) | query2doc + Hybrid 7:3 + jina-reranker-m0 | Judge 4.067 |

## 종합 보고서

- [`docs/PAPER.md`](docs/PAPER.md) — 논문 형식 요약: Abstract / Methodology / Results / Discussion / Conclusion
- [`docs/REPORT.md`](docs/REPORT.md) — 단계별 deep-dive와 통합 분석
- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) — 전체 실험 로드맵
- [`docs/cost-report.md`](docs/cost-report.md) — API 비용 분석
- [`docs/model-inventory.md`](docs/model-inventory.md) — 평가 대상 모델 카탈로그
- [`docs/experiment-design.md`](docs/experiment-design.md) — 실험 방법론
- [`docs/generation_hyperparameters.md`](docs/generation_hyperparameters.md) — LLM 생성 파라미터
- [`docs/MISTAKES_LEARNED.md`](docs/MISTAKES_LEARNED.md) — 실험 중 발견한 함정과 교훈

## 재현 방법

### 공개 결과 데이터 불러오기

```python
from datasets import load_dataset

ds = load_dataset("BAEM1N/Korean-RAG-LLM-Judge-Benchmark", "consolidated")
```

### 벤치마크 재현

이 저장소에는 원문 PDF, ground truth, 대용량 실험 결과가 포함되어 있지 않습니다. 먼저 아래 데이터를 받아 `data/`에 배치한 뒤 실행하세요.

- 원문 코퍼스: [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
- 공개 결과 데이터: [BAEM1N/Korean-RAG-LLM-Judge-Benchmark](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark)

```bash
# Stage 1-4-2 단변량
python scripts/bench_loader_extended.py --loaders all
python scripts/bench_parser_extended.py --strategies all
python scripts/bench_retriever.py --strategies all
python scripts/bench_reranker.py --rerankers all

# Stage 4-1 PreR
python scripts/bench_pre_retriever.py --strategies all

# Stage 5 axis-wise
python scripts/bench_e2e_axis_wise.py --axes A B C

# Stage 6 Cartesian
python scripts/cartesian/retrieval_matrix.py
python scripts/cartesian/rerank_apply.py --device m5
python scripts/cartesian/rerank_apply.py --device amd
python scripts/cartesian/run_gen_judge.py --workers-gen 100 --workers-judge 100
```

## 인프라

| 장비 | 사양 |
|---|---|
| DGX Spark | GB10, 128GB |
| HP Z2 Mini | AI 395+, 128GB |
| MacBook Pro | M5 Max, 128GB |

## 라이선스

- 코드: MIT
- 데이터: allganize 원본 라이선스 준수
