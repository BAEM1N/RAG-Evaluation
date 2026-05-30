# Cartesian PreR × R × PostR Pipeline (Task #37)

384 configs end-to-end 평가 (8 PreR × 6 R × 8 PostR) with multi-device reranker dispatch + massive parallel gen+judge.

## 파이프라인 단계

```
1. retrieval_matrix.py      → 48 (PreR×R) top-20 retrievals (.npy)
                              [M5 Max, gemma-embed-300m + BM25-KIWI, 약 5-10분]
2-a. rerank_apply.py --device m5  → M5 Max에서 4종 reranker × 48 = 192 outputs
2-b. rerank_apply.py --device amd → HP Z2 Mini에서 3종 reranker × 48 = 144 outputs
   no_rerank baseline = 48 outputs (자동 처리)
   둘 다 합쳐 384 top-5 결과 .npy
                              [M5 ~4h, AMD ~5-6h parallel]
3. run_gen_judge.py         → 384 × 300 gen + 384 × 1200 judge (GPT-5.4)
                              [100-worker pool, ~1.5h]
```

## 디바이스 라우팅 근거

| Reranker | Device | 근거 (per 6000 pairs) |
|---|---|---|
| bge-reranker-v2-m3 | M5 | 114s vs AMD 345s (3× 빠름, XLM-R) |
| bge-reranker-v2-m3-ko | M5 | 110s vs AMD 347s (3× 빠름) |
| ko-reranker | M5 | 96s vs AMD 347s (3.6× 빠름) |
| bge-reranker-large | M5 | XLM-R-large legacy, M5 친화 추정 |
| qwen3-reranker-0.6b | AMD | 193s vs M5 242s (1.25× 빠름) |
| mxbai-rerank-base-v2 | AMD | 82s vs M5 123s (1.5× 빠름) |
| jina-reranker-m0 | AMD | M5 미테스트, AMD 작동 확인 |
| no_rerank | (none) | top-5 = top-20[:5] |

## Pruning (770 → 384)

**제외**:
- PreR: `step_back` (axis 3.94, MRR 0.769), `multi_query_angle` (3.96, 0.643)
- R: `bm25_whitespace` (gen 3.54, MRR 0.675)
- PostR:
  - `mxbai-rerank-large-v2` (gen 3.72 < baseline 3.87)
  - `modernReranker` (3.40)
  - `bge-reranker-v2-gemma` (3.17)
  - `pixie-spell-reranker` (2.97)
  - `qwen3-reranker-4b` (9h+ rerank alone)
  - 6 incompat (sigridjineth×2, bge-v2-minicpm, bge-v2.5-gemma2-lw, jina-v3, jina-v2-multi)

## 실행 명령

### Step 1: Retrieval matrix (M5)
```bash
cd ~/Workspace/RAG-Evaluation
python scripts/cartesian/retrieval_matrix.py --skip-existing
```

### Step 2a: M5 Max reranker apply
```bash
# On M5 Max
cd ~/Workspace/RAG-Evaluation
python scripts/cartesian/rerank_apply.py --device m5 --skip-existing
```

### Step 2b: HP Z2 Mini reranker apply
```bash
# MacBook(M5): sync results/cartesian/retrieval/*.npy to HP Z2 Mini
rsync -az results/cartesian/retrieval/ baeumai@localhost:rag-bench/repo/results/cartesian/retrieval/

# HP Z2 Mini 로 전송
ssh baeumai@localhost
cd ~/rag-bench/repo && source ~/rag-bench/.venv/bin/activate
nohup setsid python scripts/cartesian/rerank_apply.py --device amd --skip-existing > /tmp/cart_rerank.log 2>&1 < /dev/null & disown

# Back on M5, pull results
rsync -az baeumai@localhost:rag-bench/repo/results/cartesian/topk/ results/cartesian/topk/
```

### Step 3: Generation + judge (massive parallel, M5)
```bash
cd ~/Workspace/RAG-Evaluation
python scripts/cartesian/run_gen_judge.py --workers-gen 100 --workers-judge 100
```

## 비용·시간 추정

| 단계 | 시간 | 비용 |
|---|---:|---:|
| 1. Retrieval matrix | 10분 | $0 |
| 2a. M5 reranker (4 models × 48) | ~4h | $0 |
| 2b. AMD reranker (3 models × 48) | ~5-6h (M5와 병렬) | $0 |
| 3. Gen + judge (576K calls, 100 worker) | ~1.5h | ~$300 |
| **합계** | **~6-8h wallclock** | **~$300** |

## 출력 산출물

```
results/cartesian/
├── retrieval/          # top20_<prer>__<r>.npy (48 files)
├── topk/               # <prer>__<r>__<postr>.npy (384 files)
├── gen/                # <config>_<qid>.txt (115K text files)
├── judge/              # <config>_<qid>_<metric>.txt (461K text files)
└── cartesian_summary.json   # 384 configs × {MRR, Hit@k, judge_means}
```
