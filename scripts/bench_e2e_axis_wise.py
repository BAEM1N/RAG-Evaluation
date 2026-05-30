#!/usr/bin/env python3
"""End-to-End axis-wise benchmark (시나리오 F) — GPT-5.4 + LangChain.

각 axis에서 하나만 변수, 나머지는 winner 고정:
  A. PreR (10): baseline (winner) 외 9개 변형 — R=Hybrid 3:7, PostR=bge-v2-m3-ko
  B. R (7): Dense, BM25-WS, BM25-KIWI, Hybrid 7:3, 5:5, 3:7 (winner), Hybrid-WS 5:5 — PreR=baseline, PostR=bge-v2-m3-ko
  C. PostR (11): no_rerank, bge-v2-m3, bge-v2-m3-ko (winner), Qwen3-0.6B/4B, mxbai-base/large-v2, bge-gemma, ko-reranker, modernReranker, pixie — PreR=baseline, R=Hybrid 3:7

총 28 configs (winner 3축 중복 제외 27). 각 config:
  1. 300 query → retrieve top-5 chunks
  2. GPT-5.4로 답변 생성 (RAG_PROMPT)
  3. GPT-5.4로 4-metric judge (similarity/correctness/completeness/faithfulness, 1-5)

비용: 약 $25-30 (GPT-5.4 batch + cache, ~$1/config)

Usage:
  python scripts/bench_e2e_axis_wise.py --axes A --dry-run        # 1 config × 3 q
  python scripts/bench_e2e_axis_wise.py --axes A B C --pilot      # 50 q
  python scripts/bench_e2e_axis_wise.py --axes A B C              # 300 q
  python scripts/bench_e2e_axis_wise.py --axes C --configs no_rerank bge-reranker-v2-m3-ko
"""
import argparse, json, time, sys, os, re
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# .env
ROOT = Path(__file__).parent.parent
for ln in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(ROOT / "scripts"))
from eval_utils import parse_pdf_pymupdf, load_ground_truth
from llm_judge import EVAL_PROMPTS  # 4-metric rubrics

PDF_DIR = ROOT / "data/pdfs"
OUT_DIR = ROOT / "results/phase_e2e_axis_wise"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GEN_DIR = OUT_DIR / "_gen_cache"; GEN_DIR.mkdir(exist_ok=True)
JUDGE_DIR = OUT_DIR / "_judge_cache"; JUDGE_DIR.mkdir(exist_ok=True)
PRER_LLM_CACHE = ROOT / "results/phase4_1_pre_retriever/_llm_cache_gpt54"

EMBED_MODEL = "google/embeddinggemma-300m"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RRF_K = 60
TOP_K = 5
TOP_N_RERANK = 20

LLM_MODEL = "gpt-5.4"
LLM_WORKERS = int(os.environ.get("LLM_WORKERS", "30"))

RAG_PROMPT = (
    "다음 검색된 문맥을 사용하여 질문에 답하세요. "
    "답을 모르면 모른다고 하세요. 최대 3문장으로 간결하게 답하세요.\n\n"
    "질문: {question}\n\n문맥:\n{context}\n\n답변:"
)

# ── Stage 4-1 strategies (PreR) ────────────────────────────────────────
PRER_STRATEGIES = [
    "baseline", "hyde", "hyde_rrf", "query2doc",
    "multi_query_para", "multi_query_angle", "step_back",
    "decompose", "query_expansion", "query_rewrite",
]
# ── Stage 4 retrievers (R) ─────────────────────────────────────────────
R_STRATEGIES = [
    "dense", "bm25_whitespace", "bm25_kiwi",
    "hybrid_7_3", "hybrid_5_5", "hybrid_3_7", "hybrid_ws_5_5",
]
# ── Stage 4-2 rerankers (PostR) ────────────────────────────────────────
POSTR_MODELS = {
    "no_rerank": None,
    "bge-reranker-v2-m3":     "BAAI/bge-reranker-v2-m3",
    "bge-reranker-v2-m3-ko":  "dragonkue/bge-reranker-v2-m3-ko",
    "qwen3-reranker-0.6b":    "Qwen/Qwen3-Reranker-0.6B",
    "qwen3-reranker-4b":      "Qwen/Qwen3-Reranker-4B",
    "mxbai-rerank-base-v2":   "mixedbread-ai/mxbai-rerank-base-v2",
    "mxbai-rerank-large-v2":  "mixedbread-ai/mxbai-rerank-large-v2",
    "bge-reranker-v2-gemma":  "BAAI/bge-reranker-v2-gemma",
    "ko-reranker":            "Dongjin-kr/ko-reranker",
    "modernReranker":         "naver/modernReranker",
    "pixie-spell-reranker":   "telepix/PIXIE-Spell-Reranker-Preview-0.6B",
}

# Winners (fixed when axis ≠ self)
WINNER_PRER = "baseline"
WINNER_R = "hybrid_3_7"
WINNER_POSTR = "bge-reranker-v2-m3-ko"


# ── chunk & retrieval primitives (numpy) ───────────────────────────────

def find_pdf(filename):
    for root, _dirs, files in os.walk(PDF_DIR):
        if filename in files:
            return Path(root) / filename
    return None


def build_chunks(gt_full):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks_text, meta = [], []
    for fname in sorted({g["target_file_name"] for g in gt_full}):
        path = find_pdf(fname)
        if not path: continue
        for p in parse_pdf_pymupdf(str(path)):
            for part in splitter.split_text(p["text"]):
                part = part.strip()
                if len(part) >= 30:
                    chunks_text.append(part)
                    meta.append({"file": fname, "page": p["page"]})
    return chunks_text, meta


def kiwi_tokenize(text):
    if not hasattr(kiwi_tokenize, "_k"):
        from kiwipiepy import Kiwi
        kiwi_tokenize._k = Kiwi()
    keep = ("N", "V", "X", "S", "M")
    return [t.form for t in kiwi_tokenize._k.tokenize(text) if t.tag.startswith(keep)]


def whitespace_tokenize(text):
    cleaned = re.sub(r"[^\w가-힣\s]", " ", text)
    return [t for t in cleaned.lower().split() if t]


_dense_model = None
_chunk_emb = None
_bm25_kiwi = None
_bm25_ws = None
_kiwi_chunk_tokens = None
_ws_chunk_tokens = None


def get_dense_model():
    global _dense_model
    from sentence_transformers import SentenceTransformer
    if _dense_model is None:
        print(f"  loading dense model {EMBED_MODEL}...", flush=True)
        _dense_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    return _dense_model


def precompute_indices(chunks_text):
    global _chunk_emb, _bm25_kiwi, _bm25_ws, _kiwi_chunk_tokens, _ws_chunk_tokens
    from rank_bm25 import BM25Okapi
    model = get_dense_model()
    print(f"  embedding {len(chunks_text)} chunks...", flush=True)
    _chunk_emb = model.encode(chunks_text, batch_size=16, show_progress_bar=True,
                              normalize_embeddings=True, convert_to_numpy=True)
    print(f"  BM25-KIWI tokenize + index...", flush=True)
    _kiwi_chunk_tokens = [kiwi_tokenize(t) for t in chunks_text]
    _bm25_kiwi = BM25Okapi(_kiwi_chunk_tokens)
    print(f"  BM25-whitespace tokenize + index...", flush=True)
    _ws_chunk_tokens = [whitespace_tokenize(t) for t in chunks_text]
    _bm25_ws = BM25Okapi(_ws_chunk_tokens)


def retrieval_topk(prer_strategy, r_strategy, postr_key, queries, qids, gt_qid_idx, top_k=TOP_K):
    """For each query: compute top_k chunk indices using (PreR, R, PostR) config."""
    n_q = len(queries)
    out = np.zeros((n_q, top_k), dtype=np.int64)
    model = get_dense_model()

    # 1. PreR — get query variants per question
    def get_prer_inputs(qid, q):
        """returns list of (dense_query, bm25_query) tuples to fuse."""
        path_for = lambda key: PRER_LLM_CACHE / f"{key}_{qid}.json"
        def load(key):
            p = path_for(key)
            return json.load(open(p))["output"] if p.exists() else None
        if prer_strategy == "baseline":
            return [(q, q)]
        if prer_strategy == "hyde":
            ans = load("hyde") or q
            return [(ans, q)]
        if prer_strategy == "hyde_rrf":
            ans = load("hyde") or q
            return [(q, q), (ans, q)]
        if prer_strategy == "query2doc":
            doc = load("query2doc") or q
            return [(f"{q} {doc}", f"{q} {doc}")]
        if prer_strategy == "multi_query_para":
            out_text = load("multiq_para")
            variants = parse_lines(out_text, 3, q) if out_text else [q]*3
            return [(q, q)] + [(v, v) for v in variants]
        if prer_strategy == "multi_query_angle":
            out_text = load("multiq_angle")
            variants = parse_lines(out_text, 3, q) if out_text else [q]*3
            return [(q, q)] + [(v, v) for v in variants]
        if prer_strategy == "step_back":
            abs_q = load("stepback") or q
            return [(q, q), (abs_q, abs_q)]
        if prer_strategy == "decompose":
            out_text = load("decompose")
            subs = parse_lines(out_text, 3, q) if out_text else [q]*3
            return [(q, q)] + [(s, s) for s in subs]
        if prer_strategy == "query_expansion":
            out_text = load("expand") or ""
            kws = parse_keywords(out_text)
            expanded = f"{q} {' '.join(kws)}" if kws else q
            return [(expanded, expanded)]
        if prer_strategy == "query_rewrite":
            rewritten = (load("rewrite") or q).split("\n")[0].strip() or q
            return [(rewritten, rewritten)]
        raise ValueError(prer_strategy)

    # 2. R — given (dense_q, bm25_q), compute hybrid scores
    def hybrid_scores(dq, bq):
        q_emb = model.encode([dq], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sims = (q_emb @ _chunk_emb.T)[0]
        dense_rank = np.argsort(-sims)
        n_c = len(sims)
        rank_d = np.empty(n_c, dtype=np.int64)
        rank_d[dense_rank] = np.arange(n_c)
        if r_strategy == "dense":
            return -rank_d.astype(np.float64)  # higher = better
        # BM25
        if "kiwi" in r_strategy or "hybrid_" in r_strategy and r_strategy != "hybrid_ws_5_5":
            bm25 = _bm25_kiwi
            bm25_q = kiwi_tokenize(bq)
        else:
            bm25 = _bm25_ws
            bm25_q = whitespace_tokenize(bq)
        bm25_scores = bm25.get_scores(bm25_q)
        bm25_rank = np.argsort(-bm25_scores)
        rank_b = np.empty(n_c, dtype=np.int64); rank_b[bm25_rank] = np.arange(n_c)
        if r_strategy == "bm25_whitespace" or r_strategy == "bm25_kiwi":
            return -rank_b.astype(np.float64)
        # Hybrid RRF
        if r_strategy == "hybrid_7_3": w_d, w_b = 0.7, 0.3
        elif r_strategy == "hybrid_5_5": w_d, w_b = 0.5, 0.5
        elif r_strategy == "hybrid_3_7": w_d, w_b = 0.3, 0.7
        elif r_strategy == "hybrid_ws_5_5": w_d, w_b = 0.5, 0.5
        else: raise ValueError(r_strategy)
        return w_d / (RRF_K + rank_d + 1) + w_b / (RRF_K + rank_b + 1)

    # 3. PostR — rerank top-N
    reranker_model = POSTR_MODELS.get(postr_key)
    rerank_ce = None
    if reranker_model is not None:
        print(f"  loading reranker {reranker_model}...", flush=True)
        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            rerank_ce = HuggingFaceCrossEncoder(model_name=reranker_model, model_kwargs={"trust_remote_code": True})
            def _score_pairs(pairs): return np.array(rerank_ce.score(pairs))
            rerank_score_fn = _score_pairs
        except Exception as e:
            print(f"  LC wrapper failed ({e}), trying sentence_transformers...", flush=True)
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(reranker_model, trust_remote_code=True)
            rerank_score_fn = lambda pairs: np.array(ce.predict(pairs, batch_size=16))

    # ── execute per query ─────────────────────────────────────────────
    n_top_for_rerank = TOP_N_RERANK if rerank_ce or reranker_model else top_k
    print(f"  retrieving for {n_q} queries (n_top={n_top_for_rerank})...", flush=True)
    pre_rerank_topk = np.zeros((n_q, n_top_for_rerank), dtype=np.int64)
    for i, (qid, q) in enumerate(zip(qids, queries)):
        inputs = get_prer_inputs(qid, q)
        score_list = [hybrid_scores(dq, bq) for dq, bq in inputs]
        fused = np.zeros_like(score_list[0])
        for s in score_list: fused += s
        top = np.argpartition(-fused, n_top_for_rerank)[:n_top_for_rerank]
        pre_rerank_topk[i] = top[np.argsort(-fused[top])]
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_q}", flush=True)

    # Rerank
    if reranker_model:
        print(f"  reranking top-{n_top_for_rerank}...", flush=True)
        # Build all (q, chunk) pairs
        pairs = []; pair_origin = []
        for i in range(n_q):
            for r in range(n_top_for_rerank):
                idx = pre_rerank_topk[i, r]
                pairs.append([queries[i], _chunks_text[idx]])
                pair_origin.append((i, idx))
        t = time.time()
        try:
            scores = rerank_score_fn(pairs)
        except Exception as e:
            print(f"  rerank failed: {e}, falling back to no_rerank")
            scores = None
        print(f"  rerank scored {len(pairs)} pairs in {time.time()-t:.1f}s", flush=True)
        if scores is not None:
            per_q_scores = [[] for _ in range(n_q)]
            per_q_idx = [[] for _ in range(n_q)]
            for (i, idx), s in zip(pair_origin, scores):
                per_q_scores[i].append(s)
                per_q_idx[i].append(idx)
            for i in range(n_q):
                order = np.argsort(-np.array(per_q_scores[i]))[:top_k]
                out[i] = [per_q_idx[i][j] for j in order]
        else:
            out = pre_rerank_topk[:, :top_k]
    else:
        out = pre_rerank_topk[:, :top_k]
    return out


def parse_lines(text, n, fallback):
    if not text: return [fallback]*n
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in lines]
    lines = [ln for ln in lines if len(ln) > 2][:n]
    while len(lines) < n: lines.append(fallback)
    return lines


def parse_keywords(text):
    if not text: return []
    raw = text.replace("\n", ",").split(",")
    kws = [w.strip() for w in raw if w.strip()]
    kws = [re.sub(r"^[\-\*\d\.\)\s]+", "", w).strip() for w in kws]
    return [w for w in kws if 1 < len(w) < 30][:5]


# ── GPT-5.4 LangChain ────────────────────────────────────────────

_llm = None


def get_llm():
    from langchain_openai import ChatOpenAI
    global _llm
    if _llm is None:
        endpoint = os.environ["LLM_API_ENDPOINT"]
        project = os.environ["LLM_API_PROJECT"]
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=os.environ["LLM_API_KEY"],
            base_url=f"{endpoint}/api/projects/{project}/openai/v1",
            temperature=0,
            max_tokens=400,
            timeout=60,
            max_retries=3,
        )
    return _llm


# ── Generation ─────────────────────────────────────────────────────────

def gen_cache_path(config_key, qid):
    return GEN_DIR / f"{config_key}_{qid}.txt"


def generate_one(config_key, qid, question, contexts):
    p = gen_cache_path(config_key, qid)
    if p.exists():
        return p.read_text()
    from langchain_core.messages import HumanMessage
    context = "\n\n".join(contexts)
    prompt = RAG_PROMPT.format(question=question, context=context)
    llm = get_llm()
    try:
        out = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        print(f"    gen err {qid}: {e}; writing empty fallback", flush=True)
        out = ""
    p.write_text(out)
    return out


def generate_all(config_key, queries, qids, contexts_list):
    todo = []
    for i, (qid, q, ctx) in enumerate(zip(qids, queries, contexts_list)):
        if not gen_cache_path(config_key, qid).exists():
            todo.append((i, qid, q, ctx))
    if todo:
        print(f"  generating {len(todo)} answers (workers={LLM_WORKERS})...", flush=True)
        t = time.time(); done = 0
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
            futures = {ex.submit(generate_one, config_key, qid, q, ctx): qid for _, qid, q, ctx in todo}
            for fut in as_completed(futures):
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{len(todo)}  ({time.time()-t:.0f}s)", flush=True)
        print(f"  gen done in {time.time()-t:.1f}s", flush=True)
    return [gen_cache_path(config_key, qid).read_text() for qid in qids]


# ── Judge ──────────────────────────────────────────────────────────────

METRICS = ["similarity", "correctness", "completeness", "faithfulness"]


def judge_cache_path(config_key, qid, metric):
    return JUDGE_DIR / f"{config_key}_{qid}_{metric}.txt"


_int_re = re.compile(r"[1-5]")


def judge_one(config_key, qid, metric, question, target, generated):
    p = judge_cache_path(config_key, qid, metric)
    if p.exists():
        return int(p.read_text().strip())
    from langchain_core.messages import HumanMessage
    prompt = EVAL_PROMPTS[metric].format(question=question, target=target, generated=generated)
    llm = get_llm().bind(max_tokens=16)
    out = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    m = _int_re.search(out)
    score = int(m.group(0)) if m else 0
    p.write_text(str(score))
    return score


def judge_all(config_key, gt_items, qids, generated_answers):
    todo = []
    for qid, item, ans in zip(qids, gt_items, generated_answers):
        for m in METRICS:
            if not judge_cache_path(config_key, qid, m).exists():
                todo.append((qid, m, item, ans))
    if todo:
        print(f"  judging {len(todo)} (q, metric) (workers={LLM_WORKERS})...", flush=True)
        t = time.time(); done = 0
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
            futures = {
                ex.submit(judge_one, config_key, qid, m, item["question"], item["target_answer"], ans): (qid, m)
                for qid, m, item, ans in todo
            }
            for fut in as_completed(futures):
                done += 1
                if done % 100 == 0:
                    print(f"    {done}/{len(todo)}  ({time.time()-t:.0f}s)", flush=True)
        print(f"  judge done in {time.time()-t:.1f}s", flush=True)

    # Aggregate
    scores = {m: [] for m in METRICS}
    for qid in qids:
        for m in METRICS:
            scores[m].append(int(judge_cache_path(config_key, qid, m).read_text().strip()))
    return scores


# ── Driver ─────────────────────────────────────────────────────────────

def run_config(axis, name, prer, r, postr, chunks_text, meta, queries, qids, gt_items):
    config_key = f"{axis}_{name}"
    result_path = OUT_DIR / f"{config_key}.json"

    if result_path.exists():
        print(f"  [skip] {config_key} already evaluated")
        return json.load(open(result_path))

    print(f"\n=== {config_key} ===")
    print(f"  PreR={prer}, R={r}, PostR={postr}")

    # 1. Retrieve top-5
    global _chunks_text
    _chunks_text = chunks_text
    topk = retrieval_topk(prer, r, postr, queries, qids, None)

    # 2. Build contexts
    contexts_list = [[chunks_text[idx] for idx in topk[i]] for i in range(len(qids))]

    # 3. Generate
    gens = generate_all(config_key, queries, qids, contexts_list)

    # 4. Judge
    scores = judge_all(config_key, gt_items, qids, gens)

    # 5. Retrieval metrics (for reference)
    hit1 = hit5 = file_hit5 = 0
    mrr = 0.0
    for i, item in enumerate(gt_items):
        tgt_f = item["target_file_name"]
        try:
            tgt_p = int(str(item["target_page_no"]).strip().split(",")[0].strip())
        except (ValueError, AttributeError):
            tgt_p = None
        found = None; ff = False
        for rank, idx in enumerate(topk[i]):
            mm = meta[idx]
            if mm["file"] == tgt_f:
                ff = True
                if tgt_p is None or mm["page"] == tgt_p:
                    if found is None: found = rank + 1
        if found == 1: hit1 += 1
        if found and found <= 5: hit5 += 1
        if ff: file_hit5 += 1
        if found: mrr += 1.0 / found
    n = len(gt_items)

    result = {
        "axis": axis, "config": name,
        "prer": prer, "r": r, "postr": postr,
        "n_q": n,
        "retrieval": {"MRR": mrr/n, "Hit@1": hit1/n, "Hit@5": hit5/n, "File@5": file_hit5/n},
        "judge_means": {m: float(np.mean(scores[m])) for m in METRICS},
        "judge_overall_mean": float(np.mean([np.mean(scores[m]) for m in METRICS])),
        "judge_high4_ratio": {m: float(np.mean([s >= 4 for s in scores[m]])) for m in METRICS},
    }
    json.dump(result, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"  MRR={result['retrieval']['MRR']:.4f}  "
          f"judge_mean={result['judge_overall_mean']:.3f}  "
          f"sim={result['judge_means']['similarity']:.2f} corr={result['judge_means']['correctness']:.2f} "
          f"comp={result['judge_means']['completeness']:.2f} faith={result['judge_means']['faithfulness']:.2f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--dry-run", action="store_true", help="3 queries × 1 config per axis")
    ap.add_argument("--pilot", action="store_true", help="50 queries")
    ap.add_argument("--configs", nargs="*", help="Filter to specific config names")
    args = ap.parse_args()

    gt_full = load_ground_truth()
    if args.dry_run: gt = gt_full[:3]
    elif args.pilot: gt = gt_full[:50]
    else: gt = gt_full
    print(f"GT: {len(gt)} Q&A (chunks over {len(gt_full)} files)")
    print(f"LLM: {LLM_MODEL}, workers={LLM_WORKERS}")

    print("\n=== building chunks ===")
    chunks_text, meta = build_chunks(gt_full)
    precompute_indices(chunks_text)
    queries = [g["question"] for g in gt]
    qids = [f"q{i:04d}" for i in range(len(gt))]

    all_results = []
    for axis in args.axes:
        if axis == "A":
            configs = [(name, name, WINNER_R, WINNER_POSTR) for name in PRER_STRATEGIES]
        elif axis == "B":
            configs = [(name, WINNER_PRER, name, WINNER_POSTR) for name in R_STRATEGIES]
        elif axis == "C":
            configs = [(name, WINNER_PRER, WINNER_R, name) for name in POSTR_MODELS]
        if args.configs:
            configs = [c for c in configs if c[0] in args.configs]
        for name, prer, r, postr in configs:
            try:
                r_out = run_config(axis, name, prer, r, postr, chunks_text, meta, queries, qids, gt)
                all_results.append(r_out)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  CONFIG ERR {axis}_{name}: {e}")

    # Summary
    print("\n" + "=" * 110)
    print(f"{'Axis':<6}{'Config':<28}{'MRR':>7}{'judge':>8}{'sim':>6}{'corr':>6}{'comp':>6}{'faith':>7}")
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: (x["axis"], -x["judge_overall_mean"])):
        jm = r["judge_means"]
        print(f"{r['axis']:<6}{r['config']:<28}{r['retrieval']['MRR']:>7.4f}"
              f"{r['judge_overall_mean']:>8.2f}"
              f"{jm['similarity']:>6.2f}{jm['correctness']:>6.2f}{jm['completeness']:>6.2f}{jm['faithfulness']:>7.2f}")


if __name__ == "__main__":
    main()
