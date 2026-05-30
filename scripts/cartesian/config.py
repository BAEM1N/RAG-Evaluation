"""Cartesian 매트릭스 + 디바이스 라우팅 정의.

Stage 1-4-2 winner 기반 pruned cartesian:
  PreR(8) × R(6) × PostR(8) = 384 configs

Reranker 디바이스 분담 (실측 timing 기반):
  - M5 Max MPS: XLM-R 기반 reranker (bge family, ko-reranker) — 3× faster than AMD
  - HP Z2 Mini: Qwen3/MxBAI/Jina-m0 — gemma/MPS 보다 빠름

제외:
  - bm25_whitespace (검색·생성 모두 최하)
  - step_back, multi_query_angle (편차 큼)
  - mxbai-large, modernReranker, bge-gemma, pixie (PostR 중 baseline보다 나쁨)
  - qwen3-reranker-4b (단독 9h+ 소요, axis-wise 데이터로 대체)
  - sigridjineth, bge-v2-minicpm, bge-v2.5-gemma2-lw, jina-v3, jina-v2-multi (transformers v5 incompat)
"""

PRER_STRATEGIES = [
    "baseline",          # no PreR
    "hyde",              # hypothetical answer
    "hyde_rrf",          # original + HyDE RRF
    "query2doc",         # original + pseudo-doc concat
    "multi_query_para",  # 3 paraphrase variants RRF
    "decompose",         # sub-question decomposition
    "query_expansion",   # keyword expansion
    "query_rewrite",     # single rewritten query
]

R_STRATEGIES = [
    "dense",
    "bm25_kiwi",
    "hybrid_7_3",
    "hybrid_5_5",
    "hybrid_3_7",
    "hybrid_ws_5_5",
]

POSTR_MODELS = {
    "no_rerank":             (None,                                "none"),
    "bge-reranker-v2-m3":    ("BAAI/bge-reranker-v2-m3",           "m5"),    # XLM-R 568M
    "bge-reranker-v2-m3-ko": ("dragonkue/bge-reranker-v2-m3-ko",   "m5"),    # KR fine-tune
    "ko-reranker":           ("Dongjin-kr/ko-reranker",            "m5"),    # KR fine-tune (bge base)
    "bge-reranker-large":    ("BAAI/bge-reranker-large",           "m5"),    # XLM-R-large legacy
    "qwen3-reranker-0.6b":   ("Qwen/Qwen3-Reranker-0.6B",          "amd"),   # decoder-style, AMD faster
    "mxbai-rerank-base-v2":  ("mixedbread-ai/mxbai-rerank-base-v2","amd"),   # AMD faster
    "jina-reranker-m0":      ("jinaai/jina-reranker-m0",           "amd"),   # AMD only (M5 untested)
}

# Total = 8 × 6 × 8 = 384 configs
N_PRER = len(PRER_STRATEGIES)
N_R = len(R_STRATEGIES)
N_POSTR = len(POSTR_MODELS)
N_CONFIGS = N_PRER * N_R * N_POSTR
N_RETRIEVAL_MATRIX = N_PRER * N_R  # 48 unique (top-20) before reranking

# Pipeline constants (Stage 1-4-2 winners)
EMBED_MODEL = "google/embeddinggemma-300m"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RRF_K = 60
TOP_N_RERANK = 20
TOP_K_FINAL = 5

# LLM (gen + judge)
LLM_MODEL = "gpt-5.4"
LLM_WORKERS_GEN = 100   # parallel HTTP workers for generation
LLM_WORKERS_JUDGE = 100 # parallel HTTP workers for judging


def all_configs():
    """Yield (config_key, prer, r, postr) for all 384 configs."""
    for prer in PRER_STRATEGIES:
        for r in R_STRATEGIES:
            for postr in POSTR_MODELS:
                key = f"{prer}__{r}__{postr}"
                yield key, prer, r, postr


def m5_postr():
    return [k for k, (_, dev) in POSTR_MODELS.items() if dev == "m5"]


def amd_postr():
    return [k for k, (_, dev) in POSTR_MODELS.items() if dev == "amd"]


def no_rerank_postr():
    return [k for k, (_, dev) in POSTR_MODELS.items() if dev == "none"]
