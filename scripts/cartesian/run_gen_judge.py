#!/usr/bin/env python3
"""Step 3: massive parallel generation + judge for all 384 configs.

Reads top-5 from results/cartesian/topk/<config_key>.npy and runs:
  - GPT-5.4 RAG answer generation
  - GPT-5.4 4-metric judge (single integer 1-5)

Uses ONE big ThreadPoolExecutor (default 100 workers) for ALL configs combined.
Total LLM calls = 384 × 300 × (1 gen + 4 judge) = 576,000 calls.

Disk cache:
  results/cartesian/gen/<config_key>_<qid>.txt
  results/cartesian/judge/<config_key>_<qid>_<metric>.txt

Usage:
  # 먼저 .env가 LLM_API_* 변수 채워져있어야 함
  python scripts/cartesian/run_gen_judge.py --workers 100
  # 일부만:
  python scripts/cartesian/run_gen_judge.py --postr ko-reranker bge-reranker-v2-m3-ko
"""
import argparse, json, time, sys, os, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

ROOT = Path(__file__).parent.parent.parent
# .env
for ln in (ROOT/".env").read_text().splitlines() if (ROOT/".env").exists() else []:
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(ROOT / "scripts"))
from cartesian.config import (
    PRER_STRATEGIES, R_STRATEGIES, POSTR_MODELS, LLM_MODEL,
    LLM_WORKERS_GEN, LLM_WORKERS_JUDGE,
)
from llm_judge import EVAL_PROMPTS

CHUNK_CACHE = ROOT / "results/phase4_2_reranker/_chunks_cache.json"
TOPK_DIR = ROOT / "results/cartesian/topk"
GEN_DIR = ROOT / "results/cartesian/gen"; GEN_DIR.mkdir(parents=True, exist_ok=True)
JUDGE_DIR = ROOT / "results/cartesian/judge"; JUDGE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "results/cartesian"

METRICS = ["similarity", "correctness", "completeness", "faithfulness"]

RAG_PROMPT = (
    "다음 검색된 문맥을 사용하여 질문에 답하세요. "
    "답을 모르면 모른다고 하세요. 최대 3문장으로 간결하게 답하세요.\n\n"
    "질문: {question}\n\n문맥:\n{context}\n\n답변:"
)


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


def load_cache():
    d = json.load(open(CHUNK_CACHE, encoding="utf-8"))
    return d["chunks_text"], d["queries"], d["gt"]


def gen_path(config_key, qid): return GEN_DIR / f"{config_key}_{qid}.txt"
def judge_path(config_key, qid, metric): return JUDGE_DIR / f"{config_key}_{qid}_{metric}.txt"


def gen_one(config_key, qid, question, contexts):
    p = gen_path(config_key, qid)
    if p.exists():
        return p.read_text()
    from langchain_core.messages import HumanMessage
    context = "\n\n".join(contexts)
    prompt = RAG_PROMPT.format(question=question, context=context)
    try:
        out = get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        out = ""
    p.write_text(out)
    return out


_int_re = re.compile(r"[1-5]")


def judge_one(config_key, qid, metric, question, target, generated):
    p = judge_path(config_key, qid, metric)
    if p.exists():
        return int(p.read_text().strip() or "0")
    from langchain_core.messages import HumanMessage
    prompt = EVAL_PROMPTS[metric].format(question=question, target=target, generated=generated)
    try:
        out = get_llm().bind(max_tokens=16).invoke([HumanMessage(content=prompt)]).content.strip()
        m = _int_re.search(out)
        score = int(m.group(0)) if m else 0
    except Exception:
        score = 0
    p.write_text(str(score))
    return score


def compute_metrics(topk, meta_list, gt_list):
    """Retrieval metrics for one (prer, r, postr) config."""
    hit1 = hit5 = file_hit5 = 0
    mrr = 0.0
    for i, g in enumerate(gt_list):
        tgt_f = g["target_file_name"]
        try:
            tgt_p = int(str(g["target_page_no"]).strip().split(",")[0].strip())
        except (ValueError, AttributeError):
            tgt_p = None
        found = None; ff = False
        for rank, idx in enumerate(topk[i]):
            m = meta_list[idx]
            if m["file"] == tgt_f:
                ff = True
                if tgt_p is None or m["page"] == tgt_p:
                    if found is None: found = rank + 1
        if found == 1: hit1 += 1
        if found and found <= 5: hit5 += 1
        if ff: file_hit5 += 1
        if found: mrr += 1.0 / found
    n = len(gt_list)
    return {"MRR": mrr/n, "Hit@1": hit1/n, "Hit@5": hit5/n, "File@5": file_hit5/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prer", nargs="*", help="filter PreR subset")
    ap.add_argument("--r", nargs="*", help="filter R subset")
    ap.add_argument("--postr", nargs="*", help="filter PostR subset")
    ap.add_argument("--workers-gen", type=int, default=LLM_WORKERS_GEN)
    ap.add_argument("--workers-judge", type=int, default=LLM_WORKERS_JUDGE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prer_keys = args.prer or PRER_STRATEGIES
    r_keys = args.r or R_STRATEGIES
    postr_keys = args.postr or list(POSTR_MODELS.keys())

    chunks_text, queries, gt = load_cache()
    meta = json.load(open(CHUNK_CACHE))["meta"]
    qids = [f"q{i:04d}" for i in range(len(queries))]
    print(f"chunks={len(chunks_text)}, queries={len(queries)}, "
          f"configs={len(prer_keys)*len(r_keys)*len(postr_keys)}")

    # ── Collect all gen jobs ─────────────────────────────────────────
    gen_jobs = []
    config_keys = []
    for prer in prer_keys:
        for r in r_keys:
            for postr in postr_keys:
                config_key = f"{prer}__{r}__{postr}"
                config_keys.append((config_key, prer, r, postr))
                topk_path = TOPK_DIR / f"{config_key}.npy"
                if not topk_path.exists():
                    print(f"  MISSING {topk_path.name}, skip")
                    continue
                topk = np.load(topk_path)
                for i, qid in enumerate(qids):
                    if gen_path(config_key, qid).exists(): continue
                    contexts = [chunks_text[idx] for idx in topk[i]]
                    gen_jobs.append((config_key, qid, queries[i], contexts))
    print(f"\nGen jobs: {len(gen_jobs)} (after skipping cache hits)")

    if args.dry_run:
        return

    # ── Run gen in massive parallel pool ─────────────────────────────
    if gen_jobs:
        print(f"\n=== Generation: {len(gen_jobs)} calls × {args.workers_gen} workers ===")
        t = time.time(); done = 0
        with ThreadPoolExecutor(max_workers=args.workers_gen) as ex:
            futs = [ex.submit(gen_one, *j) for j in gen_jobs]
            for f in as_completed(futs):
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(gen_jobs)} ({time.time()-t:.0f}s, {done/(time.time()-t):.1f}/s)", flush=True)
        print(f"  gen done in {time.time()-t:.1f}s", flush=True)

    # ── Collect judge jobs ────────────────────────────────────────────
    judge_jobs = []
    for config_key, prer, r, postr in config_keys:
        topk_path = TOPK_DIR / f"{config_key}.npy"
        if not topk_path.exists(): continue
        for i, qid in enumerate(qids):
            ans = gen_path(config_key, qid).read_text() if gen_path(config_key, qid).exists() else ""
            for m in METRICS:
                if judge_path(config_key, qid, m).exists(): continue
                judge_jobs.append((config_key, qid, m, queries[i], gt[i]["target_file_name"], ans))
    # actually judge needs target_answer not target_file_name — load full gt
    full_gt_path = ROOT / "data/ground_truth.json"
    full_gt = json.load(open(full_gt_path, encoding="utf-8"))
    judge_jobs = []
    for config_key, prer, r, postr in config_keys:
        topk_path = TOPK_DIR / f"{config_key}.npy"
        if not topk_path.exists(): continue
        for i, qid in enumerate(qids):
            ans = gen_path(config_key, qid).read_text() if gen_path(config_key, qid).exists() else ""
            for m in METRICS:
                if judge_path(config_key, qid, m).exists(): continue
                judge_jobs.append((config_key, qid, m, queries[i], full_gt[i]["target_answer"], ans))

    if judge_jobs:
        print(f"\n=== Judge: {len(judge_jobs)} calls × {args.workers_judge} workers ===")
        t = time.time(); done = 0
        with ThreadPoolExecutor(max_workers=args.workers_judge) as ex:
            futs = [ex.submit(judge_one, *j) for j in judge_jobs]
            for f in as_completed(futs):
                done += 1
                if done % 1000 == 0:
                    print(f"  {done}/{len(judge_jobs)} ({time.time()-t:.0f}s, {done/(time.time()-t):.1f}/s)", flush=True)
        print(f"  judge done in {time.time()-t:.1f}s", flush=True)

    # ── Aggregate per-config metrics ──────────────────────────────────
    print(f"\n=== Aggregating per-config metrics ===")
    rows = []
    for config_key, prer, r, postr in config_keys:
        topk_path = TOPK_DIR / f"{config_key}.npy"
        if not topk_path.exists(): continue
        topk = np.load(topk_path)
        rmetrics = compute_metrics(topk, meta, full_gt)
        scores = {m: [] for m in METRICS}
        for i, qid in enumerate(qids):
            for m in METRICS:
                p = judge_path(config_key, qid, m)
                scores[m].append(int(p.read_text().strip() or 0) if p.exists() else 0)
        jm = {m: float(np.mean(scores[m])) for m in METRICS}
        rows.append({
            "config": config_key, "prer": prer, "r": r, "postr": postr,
            **rmetrics,
            **{f"judge_{m}": jm[m] for m in METRICS},
            "judge_mean": float(np.mean(list(jm.values()))),
        })
    out_csv = OUT_DIR / "cartesian_summary.json"
    json.dump(rows, open(out_csv, "w"), ensure_ascii=False, indent=2)
    print(f"  summary saved: {out_csv} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
