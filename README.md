# Korean RAG Evaluation Benchmark

> 한국어 RAG 파이프라인 6 stage × 단변량/Cartesian 통합 벤치마크 — **무엇이 실제로 성능을 올리는가?**

[![dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue)](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark)
[![results](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/BAEM1N/RAG-Evaluation)

---

# 🎯 한 줄 결론

## **RAG 파이프라인 최적화 > 모델 업그레이드.** 동일 GPT-5.4를 쓰면서도 RAG 파이프라인만 잘 짜면 **GPT-5.4-pro(10× 더 비싼 모델)보다 +6.0pp 더 높은 정확도**를 낸다.

| Pipeline | Generator | **Accuracy** |
|---|---|---:|
| 🥇 **Cartesian winner**: query2doc + Hybrid 7:3 + jina-reranker-m0 | **GPT-5.4** | **0.827** |
| Phase 5 단순 retrieval + 동일 GPT-5.4 | GPT-5.4 | 0.787 |
| Phase 5 + GPT-5.4-pro (10× 더 비쌈) | GPT-5.4-pro | 0.767 |
| Phase 5 Open Weights 1위 (gpt-oss_120b / kimi-k2.5) | Open Weights | 0.740 |

- 동일 GPT-5.4 + RAG 파이프라인 최적화 → **+4.0pp accuracy** (0.787 → **0.827**)
- GPT-5.4-pro로 모델 업그레이드한 것보다 **+6.0pp 더 높음**
- 즉 **돈을 모델에 쓰지 말고 retrieval/reranker/PreR 조합 탐색에 써라**

---

## 📊 데이터 / 규모

- **데이터셋**: [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO) (300 Q&A × 58 PDFs × 5 도메인)
- **6 stage 단변량 비교** (576 configs 측정값) + **384 configs cartesian** 종합 평가
- **576,000 GPT-5.4 호출** — Stage 6 cartesian gen+judge
- **평가 규칙**: 18-judge 4-metric (similarity/correctness/completeness/faithfulness) majority O — Phase 5와 동일

## 🏆 Winner pipelines

| 목적 | Pipeline | Metric |
|---|---|---|
| **Accuracy·Judge 최고** | query2doc + Hybrid 7:3 + jina-reranker-m0 | **acc 0.827 / judge 4.067** |
| **MRR·Hit@1 최고** | multi_query_para + Hybrid 5:5 + jina-reranker-m0 | **MRR 0.7874 / Hit@1 75.0%** |

```
PyMuPDFLoader → LC Recursive 300/50 → embeddinggemma-300m
→ query2doc (PreR, GPT-5.4) → Hybrid 7:3 (Dense + BM25-KIWI)
→ top-20 → jinaai/jina-reranker-m0 → top-5 → GPT-5.4 답변
```

## 🎯 누적 개선 (vs naive baseline: dense only, no rerank)

| 단계 | MRR | Hit@1 | Judge | Accuracy |
|---|---:|---:|---:|---:|
| **아무 기법 X** (baseline) | 0.6816 | 59.0% | 3.850 | — |
| + Hybrid 검색 | 0.7171 | 65.3% | 3.869 | — |
| + Reranker | 0.7697 | 74.0% | 3.916 | — |
| 🏆 **Cartesian Judge/Accuracy 최고** | 0.7630 | 71.3% | **4.067** | **0.827** |
| 🏆 **MRR 최고 조합** | **0.7874** | **75.0%** | 3.991 | 0.790 |

→ **누적 개선**: +15.5% MRR, +27.1% Hit@1, +5.6% Judge (300문제 중 48문제 추가 정답)

## 📚 단계별 보고서

| Stage | 문서 | Winner | Metric |
|---|---|---|---|
| 1. Loader | [`docs/1_loader.md`](docs/1_loader.md) | pymupdf (PyMuPDFLoader) | MRR 0.6486 |
| 2. Parser | [`docs/2_parser.md`](docs/2_parser.md) | LC Recursive 300/50 (char-based 32종 + semantic 10종 + Slumber 비교) | MRR 0.6816 (dense) / 0.7171 (hybrid) |
| 3. Embedding | [`docs/3_embedding.md`](docs/3_embedding.md) | KoE5 / embeddinggemma-300m | MRR 0.6871 |
| 4. Retriever | [`docs/4_retriever.md`](docs/4_retriever.md) | Hybrid 3:7 (Dense + BM25-KIWI) | MRR 0.7171 |
| 4-1. Pre-Retriever | [`docs/4-1_pre_retriever.md`](docs/4-1_pre_retriever.md) | query_expansion (axis-wise) | MRR 0.7783 |
| 4-2. Post-Retriever | [`docs/4-2_post_retriever.md`](docs/4-2_post_retriever.md) | dragonkue/bge-v2-m3-ko | MRR 0.7697 |
| 5-A. Phase 5 generation 모델 비교 | [`docs/5_generation.md`](docs/5_generation.md) | Open vs Closed Weights leaderboard | 46 gen × 18 judge |
| 5-B. e2e Axis-wise (Stage 1~4-2 winner 고정 후 gen+judge) | [`docs/5_generation_e2e_axis.md`](docs/5_generation_e2e_axis.md) | Hybrid 5:5 + ko-reranker | Judge 3.983 (28 configs) |
| **6. Cartesian (Full)** | [`docs/6_cartesian.md`](docs/6_cartesian.md) | **query2doc + Hybrid 7:3 + jina-m0** | **Judge 4.067** |

## 📑 종합 보고서

전체 실험 결과를 한 문서에서 보려면:
- 📄 **논문 형식**: [`docs/PAPER.md`](docs/PAPER.md) — Abstract / Introduction / Related Work / Methodology / Results / Discussion / Conclusion
- 📊 **상세 기술 보고서**: [`docs/REPORT.md`](docs/REPORT.md) — 단계별 deep-dive + 통합 분석

## 🗂 보조 문서

- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) — 전체 실험 로드맵
- [`docs/cost-report.md`](docs/cost-report.md) — API 비용 분석 (Anthropic / OpenAI / Gemini)
- [`docs/model-inventory.md`](docs/model-inventory.md) — 평가 대상 모델 카탈로그
- [`docs/experiment-design.md`](docs/experiment-design.md) — 실험 방법론
- [`docs/generation_hyperparameters.md`](docs/generation_hyperparameters.md) — LLM 생성 파라미터 설정
- [`docs/MISTAKES_LEARNED.md`](docs/MISTAKES_LEARNED.md) — 실험 진행 중 발견된 함정·교훈

## 🚀 빠른 시작

> **📂 이 레포는 소스코드·분석 보고서만 담습니다.** 실험 결과 데이터(46 generator 답변 × 18 judge 평가)는
> 용량·역할 분리를 위해 GitHub 에 올리지 않고 **HuggingFace** 로 발행합니다 —
> [BAEM1N/Korean-RAG-LLM-Judge-Benchmark](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark).
> 원문 코퍼스(PDF·문항)는 [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO).

### 본 데이터셋 활용
```python
from datasets import load_dataset
ds = load_dataset("BAEM1N/Korean-RAG-LLM-Judge-Benchmark", "consolidated")
```

### 본 벤치마크 재현
> **데이터 준비 먼저**: 스크립트는 `data/` 의 코퍼스·ground truth 를 입력으로 씁니다. 이 repo 엔 데이터를 포함하지 않으므로,
> [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO) 의 원문(PDF·문항)과
> [본 데이터셋](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark) 을 받아 `data/` 에 배치한 뒤 실행하세요.
```bash
# Stage 1-4-2 단변량 (로컬 무료)
python scripts/bench_loader_extended.py --loaders all
python scripts/bench_parser_extended.py --strategies all
python scripts/bench_retriever.py --strategies all
python scripts/bench_reranker.py --rerankers all

# Stage 4-1 PreR (GPT-5.4)
python scripts/bench_pre_retriever.py --strategies all

# Stage 5 axis-wise (GPT-5.4 gen + 4-metric judge, 28 configs)
python scripts/bench_e2e_axis_wise.py --axes A B C

# Stage 6 Cartesian (384 configs) — 분산 실행
python scripts/cartesian/retrieval_matrix.py
python scripts/cartesian/rerank_apply.py --device m5    # MacBook Pro (M5 Max)
python scripts/cartesian/rerank_apply.py --device amd   # HP Z2 Mini (AI 395+)
python scripts/cartesian/run_gen_judge.py --workers-gen 100 --workers-judge 100
```

## 🛠 인프라

| 장비 | 사양 |
|---|---|
| DGX Spark | GB10, 128GB |
| HP Z2 Mini | AI 395+, 128GB |
| MacBook Pro | M5 Max, 128GB |

비용 상세는 [`docs/cost-report.md`](docs/cost-report.md), 모델 카탈로그는 [`docs/model-inventory.md`](docs/model-inventory.md) 참조.

## 📦 산출물

- **HuggingFace 데이터셋**: [BAEM1N/Korean-RAG-LLM-Judge-Benchmark](https://huggingface.co/datasets/BAEM1N/Korean-RAG-LLM-Judge-Benchmark) — 46 generator 답변 × 18 judge 평가 (384 cartesian 결과 포함)
- **재현 스크립트**: `scripts/` — 단계별 `bench_*`, 집계 `consolidate/collect/finalize`, 발행 `build_hf_dataset`, 분석 `analyze_*`
- **요약 대시보드**: <https://rag.baeum.ai.kr> — 핵심 결론·차트 인터랙티브 탐색

## 📄 라이선스

- 코드: MIT
- 데이터: allganize 원본 라이선스 준수
