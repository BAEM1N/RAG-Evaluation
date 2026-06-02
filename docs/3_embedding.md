# Embedding — 한국어 RAG에 적합한 임베딩 모델 선정

> **데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
>
> **고정 조건**: parser=pymupdf4llm, chunking=500/100, FAISS, top-k=5 (Stage 2 winner 확정 전 측정 — 이후 stage들과 다른 베이스라인)
>
> **측정**: MRR / Hit@1 / Hit@5 / File@5
>
> **공정성 보강**: 500자 truncation 제거, harrier 계열에는 `--pooling last`, context window 8192 강제

> ⚠️ **고정 조건 변경 이력**: 본 Stage 3 측정은 Stage 1·2 winner 확정 전에 진행돼 parser=pymupdf4llm + chunk 500/100 베이스라인 사용. 이후 stage들은 pymupdf + LC Recursive 300/50 로 갱신됨. 임베딩 모델 간 상대 순위는 변화 없을 것으로 추정 (chunker 효과는 모델 전반에 균등 작용).

## 31 모델 leaderboard (API 임베딩 4종 추가 — OpenAI 2 / Gemini 2)

> **API 임베딩 추가 (2026-06-02)**: 로컬 27종에 OpenAI `text-embedding-3-large/small`, Google `gemini-embedding-001`·`gemini-embedding-2`를 동일 조건으로 추가. **모든 모델 raw text·prompt-free·query/doc 무구분**(로컬과 100% 동일 방법론). Gemini에 `RETRIEVAL_QUERY/DOCUMENT` task-type을 줘도 001은 0.644로 *소폭 하락*. 흥미롭게 **최신 `gemini-embedding-2`(0.6210)가 구버전 `001`(0.6518)보다 낮음** — 한국어 도메인에선 신버전이 더 약함.

| 순위 | 모델 | dim | MRR | Hit@1 | Hit@5 | File@5 | 비고 |
|---:|---|---:|---:|---:|---:|---:|---|
| 🥇 | **koe5** | 1024 | 0.6871 | 60.7% | 80.7% | 91.3% | 한국어 특화 |
| 🥈 | **gemma-embed-300m** | 768 | 0.6650 | 57.3% | 79.7% | 91.7% | 최고의 소형 모델 (운영) |
| 🥉 | pixie-rune-v1 | 1024 | 0.6627 | 58.7% | 76.0% | 92.0% |  |
| 4 | snowflake-arctic-ko | 1024 | 0.6612 | 58.3% | 75.0% | 91.7% | 한국어 튜닝 |
| 5 | **gemini-embed-001** | 3072 | 0.6518 | 58.7% | 74.0% | 87.0% | **API**·Google·raw (task-type 시 0.644) |
| 6 | snowflake-arctic-l-v2 | 1024 | 0.6495 | 58.3% | 73.0% | 89.0% |  |
| 7 | jina-v4-retrieval | 4096 | 0.6449 | 54.7% | 78.7% | 91.7% |  |
| 8 | nomic-embed-v2-moe | 768 | 0.6435 | 56.7% | 75.3% | 90.0% | MoE |
| 9 | kure-v1 | 1024 | 0.6267 | 54.7% | 74.3% | 91.0% | 한국어 |
| 10 | **gemini-embed-2** | 3072 | 0.6210 | 54.7% | 71.0% | 87.0% | **API**·Google·raw (최신이나 001보다 낮음) |
| 11 | harrier-0.6b | 1024 | 0.6131 | 53.3% | 70.3% | 88.7% | pooling=last |
| 12 | **openai-embed-3-large** | 3072 | 0.6016 | 52.3% | 71.7% | 84.0% | **API**·OpenAI·raw |
| 13 | granite-278m | 768 | 0.5969 | 50.3% | 72.0% | 87.3% | IBM |
| 14 | me5-large-instruct | 1024 | 0.5882 | 50.7% | 70.7% | 90.7% | Multilingual-E5 |
| 15 | qwen3-embed-4b | 4096 | 0.5850 | 48.0% | 73.0% | 89.7% |  |
| 16 | bge-m3 | 1024 | 0.5630 | 48.7% | 66.7% | 89.7% |  |
| 17 | qwen3-embed-0.6b | 1024 | 0.5564 | 46.3% | 67.0% | 87.7% |  |
| 18 | **openai-embed-3-small** | 1536 | 0.5417 | 45.3% | 66.0% | 77.7% | **API**·OpenAI·raw |
| 19 | jina-v4-code | 4096 | 0.5334 | 42.3% | 67.7% | 88.0% | 코드 특화 |
| 20 | harrier-270m | 640 | 0.5291 | 43.7% | 65.3% | 88.3% | pooling=last |
| 21 | qwen3-embed-8b | 4096 | 0.5271 | 44.3% | 64.7% | 86.3% |  |
| 22 | granite-107m | 768 | 0.4786 | 38.0% | 60.3% | 83.0% |  |
| 23 | llama-embed-nemotron-8b | 4096 | 0.4617 | 36.3% | 59.0% | 88.0% | nemotron-8b와 동일 |
| 24 | nemotron-embed-8b | 4096 | 0.4617 | 36.3% | 59.0% | 88.0% |  |
| 25 | jina-v5-small-retrieval | 1024 | 0.3898 | 31.7% | 48.3% | 74.3% |  |
| 26 | jina-code-1.5b | 1024 | 0.3248 | 23.0% | 46.3% | 82.0% | 코드 특화 |
| 27 | e5-mistral-7b | 4096 | 0.2843 | 22.7% | 36.0% | 69.3% | 영어 편향 |
| 28 | jina-v5-nano-matching | 512 | 0.1791 | 12.7% | 26.3% | 62.0% | matching 튜닝 |
| 29 | mxbai-embed-large | 1024 | 0.1157 | 8.7% | 15.7% | 38.7% | 영어 전용 |
| 30 | labse | 768 | 0.0472 | 2.7% | 8.0% | 27.3% | 구형 |
| 31 | harrier-27b | 5376 | 0.0170 | 1.0% | 2.3% | 15.7% | 한국어에 부적합 |

## 핵심 관찰

1. **한국어에서는 작은 모델이 큰 영어 모델을 이긴다**: `koe5`(1024d) > `qwen3-embed-8b`(4096d), MRR 차이 **+0.16**.
2. **대형 상용 API 임베딩도 한국어 로컬에 밀린다**: `gemini-embedding-001`(3072d, 0.6518)·`openai text-embedding-3-large`(3072d, 0.6016)가 모두 `koe5`(1024d, 0.6871)·`gemma-embed-300m`(768d, 0.6650)보다 낮음. 업계 표준 OpenAI 3-large도 300M 로컬 모델 대비 **-0.063**, koe5 대비 **-0.086**. 차원·브랜드보다 **한국어 정렬**이 지배적. Gemini가 OpenAI보다 나은 건 다국어 정렬 차이로 추정.
3. **한국어 특화 임베딩이 강세**: koe5, snowflake-arctic-ko, kure-v1, pixie-rune-v1 모두 Top 9.
4. **harrier-27b 붕괴**: 5376 차원 중 약 97%가 dead dim (variance < 0.0001). 한국어 query–document 분리 자체가 실패.
5. **중복 모델 발견**: `nemotron-embed-8b`와 `llama-embed-nemotron-8b`는 동일 아키텍처/가중치 (MRR 동일 0.4617).
6. **영어 전용 모델은 하위권**: mxbai, labse, e5-mistral 모두 한국어 RAG에서 의미 없는 수준.

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
- **상용 API 임베딩(OpenAI/Gemini)은 한국어 도메인에서 정확도 이점이 없음**: 한국어 로컬 모델 대비 낮음(Gemini-001 0.6518, OpenAI 3-large 0.6016 < gemma-300m 0.6650). 정확도 열위는 본 벤치로 확인되며, 추가로 API는 호출비·외부 의존이 따른다(비용 자체는 본 벤치 측정 대상 아님).

## 레퍼런스

- KoE5 — [HF](https://huggingface.co/nlpai-lab/KoE5), [GitHub](https://github.com/nlpai-lab/KoE5) (Jang, Son, Lee 2024)
- EmbeddingGemma 300m — [HF](https://huggingface.co/google/embeddinggemma-300m), [Google AI docs](https://ai.google.dev/gemma/docs/embeddinggemma)
- Snowflake Arctic Embed L v2 — [HF](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)
- BGE-M3 — Chen et al. 2024 [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- Jina embeddings v4 — [HF](https://huggingface.co/jinaai/jina-embeddings-v4)
- Nomic Embed Text v2 MoE — [HF](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
- OpenAI text-embedding-3 — [docs](https://platform.openai.com/docs/guides/embeddings)
- Gemini embedding (001 / 2) — [docs](https://ai.google.dev/gemini-api/docs/embeddings)
- 평가: 로컬 27 모델 `scripts/bench_phase4_parallel.py`, API 4 모델은 OpenAI/Gemini SDK로 동일 raw 조건 평가
