# Post-retriever — Cross-encoder Reranker (25종 + 시도 5종)

> **데이터셋**: allganize/RAG-Evaluation-Dataset-KO (300 Q&A × 58 PDFs)
>
> **고정 파이프라인**: PyMuPDFLoader → RecursiveCharSplitter(300, 50) → EmbeddingGemma-300m → Hybrid 3:7 (FAISS + BM25-KIWI) **top-N=20** → reranker → top-5
>
> **LangChain wrapper**: `langchain_community.cross_encoders.HuggingFaceCrossEncoder` (Fall-back: `sentence_transformers.CrossEncoder`)
>
> **인프라**: M5 Max MPS (4 모델) + AMD AI 395+ ROCm 7.13 (3+14 모델) 분산

Hybrid retrieval로 top-20 후보를 받은 뒤 cross-encoder로 (query, chunk) 쌍을 재채점해 top-5로 압축. **25개 reranker + 시도 실패 5개 + no_rerank baseline = 26 측정**.

## 1. 25 reranker leaderboard (MRR 기준)

| 순위 | 모델 | 파라미터 | KR | MRR | Hit@1 | Hit@5 | File@5 | Time |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 🥇 | **dragonkue/bge-reranker-v2-m3-ko** | bge-m3 (568M) | ✅ tune | **0.7697** | **74.0%** | **81.3%** | 92.0% | 347s |
| 🥇 | shoxa-mir/bge-reranker-v2-m3-ko | bge-m3 fine-tune | ✅ tune | **0.7697** | **74.0%** | **81.3%** | 92.0% | 347s |
| 🥈 | **jinaai/jina-reranker-m0** | 2.4B 멀티모달 | ✅ (29 lang) | 0.7631 | 72.3% | 81.7% | 92.3% | 190s |
| 🥉 | Qwen/Qwen3-Reranker-4B | 4B (2025-06) | ✅ (100+) | 0.7514 | 70.3% | 81.0% | 92.3% | 713s |
| 4 | BAAI/bge-reranker-v2-m3 | 568M | ✅ (multi) | 0.7488 | 70.0% | 81.7% | 92.3% | 345s |
| 5 | SeoJHeasdw/ktds-vue-code-search-reranker-ko | 0.6B (bge fine-tune) | ✅ (KR/코드) | 0.7439 | 70.0% | 81.0% | 92.0% | 347s |
| 6 | mixedbread-ai/mxbai-rerank-base-v2 | 0.5B (Apache) | ✅ (100+) | 0.7373 | 68.3% | 81.3% | 92.3% | **82s** |
| 7 | upskyy/ko-reranker-8k | 0.6B (8k ctx) | ✅ (KR) | 0.7343 | 68.0% | 81.3% | 92.3% | 347s |
| 8 | Dongjin-kr/ko-reranker | bge-base (KR fine-tune) | ✅ (KR) | 0.7320 | 67.3% | 81.0% | 92.3% | 347s |
| 9 | Qwen/Qwen3-Reranker-0.6B | 0.6B (2025) | ✅ (100+) | 0.7255 | 66.3% | 80.7% | 92.0% | 193s |
| 10 | upskyy/ko-reranker | 0.6B | ✅ (KR) | 0.7238 | 67.7% | 80.0% | 92.3% | 347s |
| 11 | mixedbread-ai/mxbai-rerank-large-v2 | 1.5B | ✅ (100+) | 0.7229 | 66.7% | 80.0% | 91.7% | 241s |
| — | **no_rerank (baseline)** | — | — | **0.7171** | 65.3% | 80.3% | 91.7% | 0s |
| 12 | BAAI/bge-reranker-large | 560M (XLM-R-L) | ✅ (multi) | 0.7091 | 65.0% | 80.3% | 92.0% | 347s |
| 13 | nvidia/llama-nemotron-rerank-1b-v2 | 1B (2025) | ✅ (26 lang) | 0.6966 | 63.7% | 78.7% | 91.7% | 195s |
| 14 | mncai/bge-ko-reranker-560M | 560M (XLM-R/bge-L) | ✅ (KR) | 0.6341 | 55.7% | 75.0% | 92.3% | **64s** |
| 15 | BAAI/bge-reranker-base | 278M (XLM-R-base) | ✅ (multi) | 0.6339 | 53.7% | 77.0% | 91.0% | 100s |
| 16 | kkresearch/bge-reranker-v2-m3-korean-finance | 0.6B (금융 tune) | ✅ (KR 금융) | 0.5588 | 46.3% | 70.7% | 89.0% | 346s |
| 17 | Alibaba-NLP/gte-reranker-modernbert-base | 149M ModernBERT | ✅ (70+) | 0.5318 | 43.0% | 70.0% | 88.7% | 416s |
| 18 | naver/modernReranker | ModernBERT-L distill | 부분 | 0.5178 | 41.3% | 68.7% | 91.3% | 176s |
| 19 | BAAI/bge-reranker-v2-gemma | ~2B Gemma | ⚠️ | 0.4152 | 28.0% | 63.7% | 87.7% | **2,890s** |
| 20 | telepix/PIXIE-Spell-Reranker-Preview-0.6B | 0.6B | ✅ (KR/EN) | 0.3276 | 21.0% | 53.0% | 81.3% | 764s |
| 21 | naver/xprovence-reranker-bgem3-v2 | 0.6B (bge-m3 + Provence) | ✅ | 00 | 16.3% | 40.0% | 63.0% | 345s |
| 22 | js2jang/reranker_ko_qnli | 11.7M (MiniLM) | ✅ (KR) | 0.1780 | 9.0% | 34.3% | 76.3% | **31s** |
| 23 | naver/xprovence-reranker-bgem3-v1 | 0.6B | ✅ | 0.0496 | 2.3% | 10.3% | 64.0% | 347s |

### 시도했으나 실패한 모델 (transformers v5 호환성)

| 모델 | 원인 |
|---|---|
| jinaai/jina-reranker-v3 | `tensor reshape [32, 0, -1, 128]` 에러 (모델 자체 버그) |
| jinaai/jina-reranker-v2-base-multilingual | `create_position_ids_from_input_ids` deprecated in transformers v5 |
| sigridjineth/ko-reranker-v1 | 동일 (xlm_roberta API) |
| sigridjineth/ko-reranker-v1.1 | positional embedding 321 토큰 상한 초과 |
| BAAI/bge-reranker-v2-minicpm-layerwise | `is_torch_fx_available` deprecated |
| BAAI/bge-reranker-v2.5-gemma2-lightweight | `Gemma2FlashAttention2` deprecated |
| Alibaba-NLP/gte-multilingual-reranker-base | positional embedding 상한 초과 |

→ 호환 가능 모델로 25종 평가 완료.

## 2. 핵심 관찰

### (1) **한국어 fine-tune이 절대 우세**

| 카테고리 | 1위 모델 | MRR |
|---|---|---:|
| 🥇 **한국어 fine-tune** | dragonkue / shoxa-mir bge-reranker-v2-m3-ko | **0.7697** |
| 🥈 **2025 SOTA 멀티링구얼 (큰 모델)** | jinaai/jina-reranker-m0 (2.4B) | 0.7631 |
| 🥉 **2025 SOTA Qwen3** | Qwen/Qwen3-Reranker-4B | 0.7514 |

- 한국어 fine-tune 두 모델이 **모든 일반 multilingual SOTA를 압도** (+0.66pp ~ +1.83pp)
- shoxa-mir와 dragonkue는 같은 base + 비슷한 KR 데이터로 fine-tune → 결과 완전 동일
- 가장 비싼 모델(Qwen3-4B, jina-m0)보다 0.6B KR-tune이 우수

### (2) **dragonkue/bge-v2-m3-ko 동순위 변종 발견**

dragonkue와 shoxa-mir의 bge-reranker-v2-m3-ko가 **MRR 0.7697로 완전히 동일** — 같은 base model (BAAI/bge-reranker-v2-m3) + 비슷한 한국어 데이터로 독립 fine-tune. 둘 다 winner 후보로 권장 가능.

### (3) **2025 SOTA의 가성비 — Qwen3-Reranker 시리즈**

| Qwen3-Reranker | MRR | Time | 비용/효과 |
|---|---:|---:|---|
| 4B | 0.7514 | 713s | 비효율 |
| 0.6B | 0.7255 | 193s | 합리적 |

- 4B → 0.6B: MRR −2.59pp, 시간 −73% — **0.6B 가성비 우수**
- 그러나 둘 다 KR-tune 0.6B에 못 미침

### (4) **신규 한국어 모델 — 다양성 vs 품질**

본 벤치에서 신규 발견된 한국어 fine-tune reranker 8종 중:

| 모델 | MRR | 평가 |
|---|---:|---|
| shoxa-mir/bge-reranker-v2-m3-ko | **0.7697** | dragonkue tie, 최고 |
| SeoJHeasdw/ktds-vue-code-search-reranker-ko | 0.7439 | KT DS Vue 코드용이지만 일반 RAG도 잘함 |
| upskyy/ko-reranker-8k | 0.7343 | 8K context, 큰 chunk 친화 |
| Dongjin-kr/ko-reranker | 0.7320 | AWS 커뮤니티 메인테이너 |
| upskyy/ko-reranker | 0.7238 | 기본 4K |
| mncai/bge-ko-reranker-560M | 0.6341 | 모델 카드 미비 — 학습 데이터 추정 어려움 |
| kkresearch/bge-reranker-v2-m3-korean-finance | 0.5588 | 금융 도메인 over-fit, 일반 RAG에선 약함 |
| js2jang/reranker_ko_qnli | 0.1780 | 11.7M 너무 작음 |

→ **한국어 fine-tune이라고 다 좋지 않다** — 학습 데이터 도메인 / 모델 크기 중요.

### (5) **여러 모델이 baseline (no_rerank) 보다 못함**

| 모델 | MRR | vs no_rerank |
|---|---:|---:|
| no_rerank baseline | 0.7171 | (기준) |
| BAAI/bge-reranker-large | 0.7091 | **−0.0080** |
| nvidia-llama-nemotron-1b | 0.6966 | −0.0205 |
| mncai/bge-ko-reranker-560M | 0.6341 | −0.0830 |
| ... (이하 baseline 미달) | | |

→ **rerank 안 한 게 나은 모델 11/25 (44%)**. 모델 선택 신중해야.

### (6) **속도 vs 정확도 trade-off**

| 카테고리 | 모델 | MRR | Time |
|---|---|---:|---:|
| 🏆 가성비 winner | mxbai-rerank-base-v2 | 0.7373 | **82s** |
| 🏆 정확도 winner | dragonkue/bge-v2-m3-ko | **0.7697** | 347s |
| 🚫 비효율 | bge-reranker-v2-gemma | 0.4152 | **2,890s** ❌ |
| 🚫 비효율 | telepix PIXIE | 0.3276 | 764s ❌ |

### (7) **xprovence 시리즈 catastrophic**

naver/xprovence-reranker-bgem3-v1/v2 — context pruning을 위한 모델인데 RAG reranking 용도로 쓰니 점수 거의 0. **용도 misuse 사례**.

## 3. 최종 선택

**메인 운영**: `dragonkue/bge-reranker-v2-m3-ko` (또는 `shoxa-mir` 변종)
- MRR 0.7697, Hit@1 74.0%, baseline 대비 +5.26pp / +12.93% 상대
- 347초 / 300q × top-20 = 0.39s/query 추론
- bge-m3 백본 + 한국어 fine-tune, MIT 라이선스

**가성비 대안**: `mixedbread-ai/mxbai-rerank-base-v2`
- MRR 0.7373 (−3.24pp vs winner), **82초** (4.2× 빠름)
- 0.5B Apache 2.0, 작은 인프라 친화

**최고 정확도 (비용 무관)**: `jinaai/jina-reranker-m0` (cartesian gen winner)
- MRR 0.7631, 2.4B 멀티모달, 190s
- Stage 6 cartesian에서 generation quality 1위 → `query2doc + jina-m0` 조합 최강

## 4. Reranker 외 Post-retrieval 전략 (별도 실험 예정)

| 전략 | 설명 | 레퍼런스 |
|---|---|---|
| MMR (Maximum Marginal Relevance) | 다양성 + 관련성 균형 | Carbonell & Goldstein 1998 [ACL X98-1025](https://aclanthology.org/X98-1025/) |
| Multi-reranker RRF | 2+ reranker 앙상블 | Cormack 2009 [DOI](https://doi.org/10.1145/1571941.1572114) |
| Score normalization (zELO) | min-max / Platt / Elo 보정 | [arXiv:2509.12541](https://arxiv.org/abs/2509.12541) |
| DSLR (sentence-level rerank) | chunk → 문장 분해 → 무관 문장 필터 | [arXiv:2407.03627](https://arxiv.org/abs/2407.03627) |
| Late interaction (ColBERTv2) | 토큰 단위 MaxSim | [arXiv:2112.01488](https://arxiv.org/abs/2112.01488) |
| LLM-as-reranker | RankGPT/RankZephyr/RankLLaMA | RankZephyr [arXiv:2312.02724](https://arxiv.org/abs/2312.02724) |
| Contextual compression | LongLLMLingua / LLMLingua-2 | [arXiv:2310.06839](https://arxiv.org/abs/2310.06839) |
| ChunkRAG | LLM-chunk 필터링 | [arXiv:2410.19572](https://arxiv.org/abs/2410.19572) |

## 5. 권장 운영 구성

```
[질문]
  → [pymupdf + LC Recursive 300/50]
  → [Hybrid 3:7 (EmbeddingGemma-300m + BM25-KIWI), top-20]
  → [dragonkue/bge-reranker-v2-m3-ko, top-5]
  → [LLM 답변 생성]
```

전체 파이프라인 MRR: **0.7697** (단순 dense 0.6816 대비 **+12.93%**)
또는 **`jina-reranker-m0`** 로 교체 시 generation quality 추가 +5.6% (Stage 6 cartesian winner)

## 6. Reranker 모델 레퍼런스 (HuggingFace)

| 모델 | HF URL |
|---|---|
| dragonkue/bge-reranker-v2-m3-ko | [link](https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko) |
| shoxa-mir/bge-reranker-v2-m3-ko | [link](https://huggingface.co/shoxa-mir/bge-reranker-v2-m3-ko) |
| jinaai/jina-reranker-m0 | [link](https://huggingface.co/jinaai/jina-reranker-m0) |
| Qwen/Qwen3-Reranker-0.6B / 4B | [0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) · [4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) |
| BAAI/bge-reranker-v2-m3 | [link](https://huggingface.co/BAAI/bge-reranker-v2-m3) ([Chen et al. 2024 arXiv:2402.03216](https://arxiv.org/abs/2402.03216)) |
| BAAI/bge-reranker-large / base | [large](https://huggingface.co/BAAI/bge-reranker-large) · [base](https://huggingface.co/BAAI/bge-reranker-base) |
| BAAI/bge-reranker-v2-gemma | [link](https://huggingface.co/BAAI/bge-reranker-v2-gemma) |
| Dongjin-kr/ko-reranker | [link](https://huggingface.co/Dongjin-kr/ko-reranker) |
| upskyy/ko-reranker · ko-reranker-8k | [base](https://huggingface.co/upskyy/ko-reranker) · [8k](https://huggingface.co/upskyy/ko-reranker-8k) |
| mixedbread-ai/mxbai-rerank-base-v2 / large-v2 | [base](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) · [large](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) |
| nvidia/llama-nemotron-rerank-1b-v2 | [link](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) |
| SeoJHeasdw/ktds-vue-code-search-reranker-ko | [link](https://huggingface.co/SeoJHeasdw/ktds-vue-code-search-reranker-ko) |
| naver/modernReranker | [link](https://huggingface.co/naver/modernReranker) |
| naver/xprovence-reranker-bgem3-v1 / v2 | [v1](https://huggingface.co/naver/xprovence-reranker-bgem3-v1) · [v2](https://huggingface.co/naver/xprovence-reranker-bgem3-v2) |
| Alibaba-NLP/gte-reranker-modernbert-base | [link](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) |
| telepix/PIXIE-Spell-Reranker-Preview-0.6B | [link](https://huggingface.co/telepix/PIXIE-Spell-Reranker-Preview-0.6B) |
| mncai/bge-ko-reranker-560M | [link](https://huggingface.co/mncai/bge-ko-reranker-560M) |
| kkresearch/bge-reranker-v2-m3-korean-finance | [link](https://huggingface.co/kkresearch/bge-reranker-v2-m3-korean-finance) |
| js2jang/reranker_ko_qnli | [link](https://huggingface.co/js2jang/reranker_ko_qnli) |
