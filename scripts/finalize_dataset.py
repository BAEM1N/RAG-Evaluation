#!/usr/bin/env python3
"""Final dataset cleanup + Q4 GPT extension.

1. Drop duplicate judges (nemotron-120b, qwen3.6-35b-a3b — these are aliases)
2. Parse Q4 GPT raw (OpenAI Responses API) → 4 judges × 34 API cands × 300 q × 4 metrics
3. Rebuild wide format for all 18 judges
4. Recompute consolidated (majority voting)
"""
import json, glob, re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
GT_FILE = ROOT / "data/ground_truth_filtered.json"
PROV_DIR = ROOT / "results/phase5_exp_b_provider"
LLM_DIR = ROOT / "results/phase5_exp_b_llm"
CONS_DIR = ROOT / "results/phase5_judge_consolidated"
FLAGSHIP_RAW_LOCAL = ROOT / "_backups/phase5_judge_flagship_raw"
FLAGSHIP = ROOT / "results/phase5_judge_flagship"
OUT_DIR = ROOT / "publish"
OUT_DIR.mkdir(exist_ok=True)

DROP_JUDGES = {"nemotron-120b", "qwen3.6-35b-a3b"}
GPT_RAW_Q4 = {
    "gpt-5.4": [FLAGSHIP_RAW_LOCAL / "_raw_openai_5.4_q4.jsonl",
                FLAGSHIP / "q4_openai_54_single_retry.jsonl"],
    "gpt-5.4-mini": [FLAGSHIP_RAW_LOCAL / "_raw_openai_5.4_mini_q4.jsonl",
                     FLAGSHIP / "q4_openai_54mini_single_retry.jsonl"],
    "gpt-5.4-nano": [FLAGSHIP_RAW_LOCAL / "_raw_openai_5.4_nano_q4.jsonl",
                     FLAGSHIP_RAW_LOCAL / "_raw_openai_q4_nano.jsonl",
                     FLAGSHIP / "q4_openai_54nano_single_retry.jsonl"],
    "gpt-5.5": [FLAGSHIP_RAW_LOCAL / "_raw_openai_5.5_q4.jsonl",
                FLAGSHIP / "_raw_openai_5.5_q4_retry.jsonl",
                FLAGSHIP / "_raw_openai_5.5_q4_retry2.jsonl",
                FLAGSHIP / "_raw_openai_5.5_q4_retry3_sync.jsonl",
                FLAGSHIP / "q4_openai_55_single_retry.jsonl"],
}
METRICS = ["similarity", "correctness", "completeness", "faithfulness"]


def parse_openai_q4(judge):
    """Parse Q4 GPT raw → {(cand_norm, qidx, metric): score}"""
    scores = {}
    for path in GPT_RAW_Q4[judge]:
        if not path.exists(): continue
        for line in open(path):
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
            except:
                continue
            cid = d.get("custom_id", "")
            m = re.match(r"(.+?)__q(\d{3})__(\w+)$", cid)
            if not m: continue
            cand, qidx, metric = m.group(1), int(m.group(2)), m.group(3)
            cand_norm = cand.replace("/", "_")
            # restore dots: _ between digits (e.g. gpt-5_4 → gpt-5.4, v2_5 → v2.5)
            cand_norm = re.sub(r"(\d)_(\d)", r"\1.\2", cand_norm)
            # extract text
            body = d.get("response", {}).get("body", {})
            text = ""
            for o in body.get("output", []):
                for c in o.get("content", []):
                    if c.get("type") == "output_text":
                        text += str(c.get("text", ""))
            # parse last 1-5
            mm = re.findall(r"\b([1-5])\b", text)
            if mm:
                scores[(cand_norm, qidx, metric)] = int(mm[-1])
    return scores


def compute_ox(votes):
    """4 metric votes → O/X (≥2 score 4)"""
    if not votes or len(votes) < 4:
        return ""
    return "O" if sum(1 for v in votes.values() if v >= 4) >= 2 else "X"


def load_cand_answers():
    answers = {}
    for base_dir in [PROV_DIR, LLM_DIR]:
        for f in sorted(base_dir.glob("expB__gemma-embed-300m__*")):
            if ".bak" in f.name: continue
            stem = f.name.replace("expB__gemma-embed-300m__", "")
            cand = re.sub(r"\.(jsonl?|json)$", "", stem)
            if cand.startswith("_") or cand.endswith("_retry") or "registry" in cand:
                continue
            if f.suffix == ".json":
                d = json.load(open(f))
                results = d.get("results", [])
            elif f.suffix == ".jsonl":
                results = [json.loads(l) for l in open(f) if l.strip()]
            else: continue
            by_qid = {}
            for i, r in enumerate(results):
                raw = r.get("qid") or r.get("custom_id") or ""
                mm = re.search(r"q(\d{3})", str(raw))
                idx = int(mm.group(1)) if mm else i
                by_qid[idx] = r.get("generated_answer") or r.get("answer") or r.get("output", "")
            answers[cand] = by_qid
    return answers


def load_oss_judges():
    """Load 9 surviving local judges from consolidated."""
    results = {}
    for f in sorted(CONS_DIR.glob("judge_*.json")):
        if ".bak" in f.name: continue
        m = re.match(r"judge_(.+?)_nothink__expB__gemma-embed-300m__(.+?)\.json$", f.name)
        if not m: continue
        judge, cand = m.group(1), m.group(2)
        if judge in DROP_JUDGES: continue
        d = json.load(open(f))
        results.setdefault(judge, {})[cand] = [s.get("result", "X") for s in d.get("scores", [])]
    return results


def load_api_judges(cand_set):
    """Load 5 full-coverage API judges (Claude + Gemini) from long format."""
    long_path = Path("./hf-export/data/judge_scores.parquet")
    df = pd.read_parquet(long_path)
    api_judges = ["claude-opus-4-7", "claude-sonnet-4-6",
                  "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
                  "gemini-3.1-pro-preview"]
    results = {}
    for judge in api_judges:
        sub = df[df["judge_id"] == judge]
        pivot = sub.pivot_table(
            index=["qid", "cand_id"], columns="metric", values="score", aggfunc="first"
        ).reset_index()
        pivot[METRICS] = pivot[METRICS].fillna(0).astype(int)
        ox_per_cand = {}
        for _, r in pivot.iterrows():
            cand = r["cand_id"].replace("/", "_")
            qidx = int(r["qid"].replace("q", ""))
            if cand not in cand_set: continue
            votes = {m: r[m] for m in METRICS}
            ox_per_cand.setdefault(cand, [""] * 300)
            ox_per_cand[cand][qidx] = compute_ox(votes)
        results[judge] = ox_per_cand
    return results


def load_gpt_q4(cand_set):
    """Q4 GPT raw → ox per (judge, cand, qidx)"""
    results = {}
    for judge in GPT_RAW_Q4:
        scores = parse_openai_q4(judge)
        # build votes per (cand, qidx)
        votes_by_cand = {}
        for (cand, qidx, metric), s in scores.items():
            if cand not in cand_set: continue
            votes_by_cand.setdefault((cand, qidx), {})[metric] = s
        ox_per_cand = {}
        for (cand, qidx), votes in votes_by_cand.items():
            ox_per_cand.setdefault(cand, [""] * 300)
            ox_per_cand[cand][qidx] = compute_ox(votes)
        results[judge] = ox_per_cand
    return results


def main():
    print("Loading...")
    gt = json.load(open(GT_FILE))
    cand_answers = load_cand_answers()
    print(f"  cand answers: {len(cand_answers)}")

    oss_judges = load_oss_judges()
    print(f"  OSS judges (dedup): {len(oss_judges)} = {sorted(oss_judges.keys())}")

    cand_set = set(cand_answers.keys())
    api_judges = load_api_judges(cand_set)
    print(f"  API full-coverage judges: {len(api_judges)} = {sorted(api_judges.keys())}")

    gpt_q4 = load_gpt_q4(cand_set)
    print(f"  GPT Q4 judges: {len(gpt_q4)}")
    for j, d in gpt_q4.items():
        evaluated = sum(sum(1 for x in arr if x in ("O", "X")) for arr in d.values())
        print(f"    {j}: {len(d)} cands, {evaluated} cells")

    # Get existing OSS Q3 GPT data from long format (for GPT × local cand)
    long_df = pd.read_parquet("./hf-export/data/judge_scores.parquet")
    for judge in GPT_RAW_Q4:
        sub = long_df[long_df["judge_id"] == judge]
        pivot = sub.pivot_table(
            index=["qid", "cand_id"], columns="metric", values="score", aggfunc="first"
        ).reset_index()
        pivot[METRICS] = pivot[METRICS].fillna(0).astype(int)
        for _, r in pivot.iterrows():
            cand = r["cand_id"].replace("/", "_")
            if cand not in cand_set: continue
            qidx = int(r["qid"].replace("q", ""))
            votes = {m: r[m] for m in METRICS}
            ox = compute_ox(votes)
            if ox:
                gpt_q4.setdefault(judge, {}).setdefault(cand, [""] * 300)
                if gpt_q4[judge][cand][qidx] == "":
                    gpt_q4[judge][cand][qidx] = ox

    print(f"\nAfter merge Q3+Q4 for GPT judges:")
    for j, d in gpt_q4.items():
        evaluated = sum(sum(1 for x in arr if x in ("O", "X")) for arr in d.values())
        print(f"  {j}: {len(d)} cands, {evaluated}/{46*300} cells ({evaluated/(46*300)*100:.0f}%)")

    cands = sorted(cand_answers.keys())
    base_cols = ["domain", "question", "target_answer", "target_file_name", "target_page_no", "context_type"]

    # Clear old publish dir
    for f in OUT_DIR.glob("*.parquet"):
        f.unlink()
    for f in OUT_DIR.glob("*.csv"):
        f.unlink()

    all_judges = sorted(oss_judges.keys()) + sorted(api_judges.keys()) + sorted(gpt_q4.keys())
    print(f"\nTotal judges: {len(all_judges)}")

    def build_df(judge_data):
        rows = []
        for i, g in enumerate(gt):
            row = {k: g.get(k) for k in base_cols}
            for cand in cands:
                row[f"{cand}_answer"] = cand_answers[cand].get(i, "")
                if isinstance(judge_data.get(cand), list):
                    row[f"{cand}_ox"] = judge_data[cand][i] if i < len(judge_data[cand]) else ""
                else:
                    arr = judge_data.get(cand, [""] * 300)
                    row[f"{cand}_ox"] = arr[i] if isinstance(arr, list) and i < len(arr) else ""
            rows.append(row)
        ordered = base_cols.copy()
        for cand in cands:
            ordered.extend([f"{cand}_answer", f"{cand}_ox"])
        return pd.DataFrame(rows)[ordered]

    # Build each judge
    for judge in sorted(oss_judges.keys()):
        df = build_df(oss_judges[judge])
        df.to_parquet(OUT_DIR / f"{judge}.parquet", index=False)
        df.to_csv(OUT_DIR / f"{judge}.csv", index=False)
        ev = sum((df[f"{c}_ox"].isin(["O", "X"])).sum() for c in cands)
        o = sum((df[f"{c}_ox"] == "O").sum() for c in cands)
        print(f"  {judge}: acc={o/ev:.3f} cov={ev/(len(cands)*300)*100:.0f}%")

    for judge in sorted(api_judges.keys()):
        df = build_df(api_judges[judge])
        df.to_parquet(OUT_DIR / f"{judge}.parquet", index=False)
        df.to_csv(OUT_DIR / f"{judge}.csv", index=False)
        ev = sum((df[f"{c}_ox"].isin(["O", "X"])).sum() for c in cands)
        o = sum((df[f"{c}_ox"] == "O").sum() for c in cands)
        print(f"  {judge}: acc={o/ev:.3f} cov={ev/(len(cands)*300)*100:.0f}%")

    for judge in sorted(gpt_q4.keys()):
        df = build_df(gpt_q4[judge])
        df.to_parquet(OUT_DIR / f"{judge}.parquet", index=False)
        df.to_csv(OUT_DIR / f"{judge}.csv", index=False)
        ev = sum((df[f"{c}_ox"].isin(["O", "X"])).sum() for c in cands)
        o = sum((df[f"{c}_ox"] == "O").sum() for c in cands)
        print(f"  {judge}: acc={o/ev:.3f} cov={ev/(len(cands)*300)*100:.0f}%")

    # Rebuild consolidated
    all_files = sorted(OUT_DIR.glob("*.parquet"))
    template = pd.read_parquet(all_files[0])[base_cols + [f"{c}_answer" for c in cands]].copy()
    matrices = {}
    for f in all_files:
        d = pd.read_parquet(f)
        for c in cands:
            matrices.setdefault(c, []).append(d[f"{c}_ox"].values)

    for c in cands:
        arr = np.array(matrices[c])
        o_count = (arr == "O").sum(axis=0)
        eval_count = ((arr == "O") | (arr == "X")).sum(axis=0)
        template[f"{c}_ox"] = ["O" if e > 0 and o > e / 2 else "X" for o, e in zip(o_count, eval_count)]

    ordered = base_cols.copy()
    for cand in cands:
        ordered.extend([f"{cand}_answer", f"{cand}_ox"])
    template = template[ordered]
    template.to_parquet(OUT_DIR / "consolidated.parquet", index=False)
    template.to_csv(OUT_DIR / "consolidated.csv", index=False)
    o = sum((template[f"{c}_ox"] == "O").sum() for c in cands)
    print(f"\nconsolidated ({len(all_files)} judges): acc={o/(len(cands)*300):.3f}")


if __name__ == "__main__":
    main()
