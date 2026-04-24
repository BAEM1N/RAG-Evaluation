#!/usr/bin/env python3
"""Aggregate Phase 5 Exp B judge results into a cross-judge leaderboard.

Input: results/phase5_judge/judge_<judge>__expB__gemma-embed-300m__<llm>.json
Output: prints markdown leaderboard + writes results/phase5_judge/LEADERBOARD.md
"""
import json, glob, re
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1]
JDIR = BASE / "results" / "phase5_judge"

# Judges known to have broken outputs (all-zero scores from parser issues)
EXCLUDE_JUDGES = {"solar-open-100b_nothink"}

j = defaultdict(dict)
for f in sorted(JDIR.glob("judge_*.json")):
    d = json.load(open(f))
    if d["scored"] != d["total"]:
        continue
    judge = f"{d['judge_model']}_{d['judge_mode']}"
    if judge in EXCLUDE_JUDGES:
        continue
    m = re.match(r"judge_.+?__expB__gemma-embed-300m__(.+)\.json", f.name)
    if not m:
        continue
    llm = m.group(1)
    acc = d.get("accuracy")
    if acc is None:
        continue
    j[llm][judge] = acc

# Auto-detect broken judges: all-zero across >=50% of their files
broken = set()
for judge in {jn for d in j.values() for jn in d}:
    accs = [d[judge] for d in j.values() if judge in d]
    if accs and sum(1 for a in accs if a == 0.0) / len(accs) >= 0.5:
        broken.add(judge)
if broken:
    print(f"Auto-excluding broken judges (all-zero ≥50%): {sorted(broken)}")
    for llm in j:
        for b in broken:
            j[llm].pop(b, None)

judges = sorted({jn for v in j.values() for jn in v})
complete_llms = [llm for llm, d in j.items() if all(jn in d for jn in judges)]

print(f"Judges with ≥1 complete result: {len(judges)}")
for jn in judges:
    n = sum(1 for v in j.values() if jn in v)
    print(f"  {jn:50s} {n}/12")
print(f"LLMs judged by ALL {len(judges)} judges: {len(complete_llms)}")

rows = []
for llm in complete_llms:
    scores = [j[llm][jn] for jn in judges]
    avg = sum(scores) / len(scores)
    rows.append((llm, scores, avg))
rows.sort(key=lambda x: -x[2])

lines = []
lines.append("# Phase 5 Exp B — LLM-as-Judge Leaderboard\n")
lines.append(f"**Retrieval**: gemma-embed-300m (FAISS, top-5)  ")
lines.append(f"**Judges**: {len(judges)} (all 300 Q&A scored, allganize methodology: 4 metrics × threshold=4 × majority vote)  ")
lines.append(f"**LLMs**: {len(complete_llms)}/12 complete\n")
lines.append("## Judges\n")
for jn in judges:
    lines.append(f"- `{jn}`")
lines.append("")
lines.append("## Cross-judge accuracy (O rate)\n")
hdr = ["Rank", "LLM"] + [jn.replace("_nothink", "")[:22] for jn in judges] + ["**Avg**"]
lines.append("| " + " | ".join(hdr) + " |")
lines.append("|" + "---|" * len(hdr))
for i, (llm, scores, avg) in enumerate(rows, 1):
    cells = [str(i), f"`{llm}`"] + [f"{s:.4f}" for s in scores] + [f"**{avg:.4f}**"]
    lines.append("| " + " | ".join(cells) + " |")

# also write raw JSON for downstream consumption
raw = {
    "judges": judges,
    "excluded": sorted(EXCLUDE_JUDGES),
    "ranking": [{"llm": llm, "scores": dict(zip(judges, scores)), "avg": avg} for llm, scores, avg in rows],
    "partial": {llm: d for llm, d in j.items() if llm not in complete_llms},
}

out_md = JDIR / "LEADERBOARD.md"
out_md.write_text("\n".join(lines) + "\n")
(JDIR / "leaderboard.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2))
print(f"\nWrote {out_md}")
print(f"Wrote {JDIR / 'leaderboard.json'}")
print()
print("\n".join(lines))
