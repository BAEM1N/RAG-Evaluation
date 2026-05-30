# Embedding — 한국어 RAG에 적합한 임베딩 모델 선정

> **데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
>
> **고정 조건**: parser=pymupdf4llm, chunking=500/100, FAISS, top-k=5 (Stage 2 winner 확정 전 측정 — 이후 stage들과 다른 베이스라인)
>
> **측정**: MRR / Hit@1 / Hit@5 / File@5
>
> **공정성 보강**: 500자 truncation 제거, harrier 계열에는 `--pooling last`, context window 8192 강제

> ⚠️ **고정 조건 변경 이력**: 본 Stage 3 측정은 Stage 1·2 winner 확정 전에 진행돼 parser=pymupdf4llm + chunk 500/100 베이스라인 사용. 이후 stage들은 pymupdf + LC Recursive 300/50 로 갱신됨. 임베딩 모델 간 상대 순위는 변화 없을 것으로 추정 (chunker 효과는 모델 전반에 균등 작용).

## 27 모델 leaderboard

| 순위 | 모델 | dim | MRR | Hit@1 | Hit@5 | File@5 | 비고 |
|---:|---|---:|---:|---:|---:|---:|---|
| 🥇 | **koe5** | 1024 | **0.6871** | 60.7% | 80.7% | 91.3% | 한국어 특화 |
| 🥈 | gemma-embed-300m | 768 | 0.6650 | 57.3% | 79.7% | 91.7% | 최고의 소형 모델 |
| 🥉 | pixie-rune-v1 | 1024 | 0.6627 | 58.7% | 76.0% | 92.0% | |
| 4 | snowflake-arctic-ko | 1024 | 0.6612 | 58.3% | 75.0% | 91.7% | 한국어 튜닝 |
| 5 | snowflake-arctic-l-v2 | 1024 | 0.6495 | 58.3% | 73.0% | 89.0% | |
| 6 | jina-v4-retrieval | 4096 | 0.6449 | 54.7% | 78.7% | 91.7% | |
| 7 | nomic-embed-v2-moe | 768 | 0.6435 | 56.7% | 75.3% | 90.0% | MoE |
| 8 | kure-v1 | 1024 | 0.6267 | 54.7% | 74.3% | 91.0% | 한국어 |
| 9 | harrier-0.6b | 1024 | 0.6131 | 53.3% | 70.3% | 88.7% | pooling=last |
| 10 | granite-278m | 768 | 0.5969 | 50.3% | 72.0% | 87.3% | IBM |
| 11 | me5-large-instruct | 1024 | 0.5882 | 50.7% | 70.7% | 90.7% | Multilingual-E5 |
| 12 | qwen3-embed-4b | 4096 | 0.5850 | 48.0% | 73.0% | 89.7% | |
| 13 | bge-m3 | 1024 | 0.5630 | 48.7% | 66.7% | 89.7% | |
| 14 | qwen3-embed-0.6b | 1024 | 0.5564 | 46.3% | 67.0% | 87.7% | |
| 15 | jina-v4-code | 4096 | 0.5334 | 42.3% | 67.7% | 88.0% | 코드 특화 |
| 16 | harrier-270m | 640 | 0.5291 | 43.7% | 65.3% | 88.3% | pooling=last |
| 17 | qwen3-embed-8b | 4096 | 0.5271 | 44.3% | 64.7% | 86.3% | |
| 18 | granite-107m | 768 | 0.4786 | 38.0% | 60.3% | 83.0% | |
| 19 | nemotron-embed-8b | 4096 | 0.4617 | 36.3% | 59.0% | 88.0% | |
| 20 | llama-embed-nemotron-8b | 4096 | 0.4617 | 36.3% | 59.0% | 88.0% | #19와 동일 |
| 21 | jina-v5-small-retrieval | 1024 | 0.3898 | 31.7% | 48.3% | 74.3% | |
| 22 | jina-code-1.5b | 1024 | 0.3248 | 23.0% | 46.3% | 82.0% | 코드 특화 |
| 23 | e5-mistral-7b | 4096 | 0.2843 | 22.7% | 36.0% | 69.3% | 영어 편향 |
| 24 | jina-v5-nano-matching | 512 | 0.1791 | 12.7% | 26.3% | 62.0% | matching 튜닝 |
| 25 | mxbai-embed-large | 1024 | 0.1157 | 8.7% | 15.7% | 38.7% | 영어 전용 |
| 26 | labse | 768 | 0.0472 | 2.7% | 8.0% | 27.3% | 구형 |
| 27 | harrier-27b | 5376 | 0.0170 | 1.0% | 2.3% | 15.7% | 한국어에 부적합 |

## 핵심 관찰

1. **한국어에서는 작은 모델이 큰 영어 모델을 이긴다**: `koe5`(1024d) > `qwen3-embed-8b`(4096d), MRR 차이 **+0.16**.
2. **한국어 특화 임베딩이 강세**: koe5, snowflake-arctic-ko, kure-v1, pixie-rune-v1 모두 Top 8.
3. **harrier-27b 붕괴**: 5376 차원 중 약 97%가 dead dim (variance < 0.0001). 한국어 query–document 분리 자체가 실패.
4. **중복 모델 발견**: `nemotron-embed-8b`와 `llama-embed-nemotron-8b`는 동일 아키텍처/가중치 (MRR 동일 0.4617).
5. **영어 전용 모델은 하위권**: mxbai, labse, e5-mistral 모두 한국어 RAG에서 의미 없는 수준.

## 공정성 보강 작업

| 항목 | 영향 |
|---|---|
| 500자 truncation 제거 | 대형 모델의 long-context 손해 완화 |
| harrier-0.6b `--pooling last` | MRR +0.094 vs mean pooling |
| harrier-270m mean pooling | last보다 marginally 우세 (0.5479 → 0.5291) |
| harrier-27b 양쪽 pooling 측정 | last 약간 우세 (0.0033 → 0.0170), 그래도 사용 불가 |
| qwen3-embed-8b ctx=8192 명시 | MRR 변화 없음, 0.5271 확정 |

## 결론

- **메인 권장**: `koe5` (한국어 RAG 1순위)
- **소형 인프라 / 본 벤치 운영**: `gemma-embed-300m` (768d, MRR 0.6650, 빠른 latency) — Stage 4 이후 모든 stage에서 사용
- **다국어 호환 필요**: `snowflake-arctic-l-v2` 또는 `jina-v4-retrieval`

## 레퍼런스

- KoE5 — [HF](https://huggingface.co/nlpai-lab/KoE5), [GitHub](https://github.com/nlpai-lab/KoE5) (Jang, Son, Lee 2024)
- EmbeddingGemma 300m — [HF](https://huggingface.co/google/embeddinggemma-300m), [Google AI docs](https://ai.google.dev/gemma/docs/embeddinggemma)
- Snowflake Arctic Embed L v2 — [HF](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)
- BGE-M3 — Chen et al. 2024 [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- Jina embeddings v4 — [HF](https://huggingface.co/jinaai/jina-embeddings-v4)
- Nomic Embed Text v2 MoE — [HF](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
- (전체 27 모델 평가 코드: `scripts/bench_phase4_parallel.py`)
