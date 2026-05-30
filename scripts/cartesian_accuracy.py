#!/usr/bin/env python3
"""Cartesian 384 configs accuracy 계산 (Phase 5 동일 규칙).

규칙: 각 질문에 대해 4 metric (sim/corr/comp/faith) 중 ≥2개가 점수 ≥4 → O, 아니면 X.
Accuracy = (O 개수) / 300.

출력: cartesian_summary.json 업데이트 + all_stages_summary 재생성.
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
JUDGE_DIR = ROOT / "results/cartesian/judge"
CART_PATH = ROOT / "results/cartesian/cartesian_summary.json"
METRICS = ["similarity", "correctness", "completeness", "faithfulness"]
THRESHOLD = 4
MAJORITY = 2  # ≥2 of 4 metrics
N_Q = 300


def main():
    # Collect per-config per-question scores
    print("Scanning judge files...", flush=True)
    cfg_q_scores = defaultdict(lambda: defaultdict(dict))  # {config: {qid: {metric: score}}}
    for f in JUDGE_DIR.glob("*.txt"):
        name = f.stem  # e.g. "baseline__dense__no_rerank_q0000_similarity"
        # split last 2 underscore-segments: qid + metric
        parts = name.rsplit("_", 2)  # ['baseline__dense__no_rerank', 'q0000', 'similarity']
        if len(parts) < 3: continue
        config, qid, metric = parts
        if metric not in METRICS: continue
        try:
            score = int(f.read_text().strip() or "0")
        except (ValueError, OSError):
            score = 0
        cfg_q_scores[config][qid][metric] = score

    print(f"Configs: {len(cfg_q_scores)}")

    # Compute accuracy per config
    accuracies = {}
    for config, q_dict in cfg_q_scores.items():
        n_ox = 0
        for qid, scores in q_dict.items():
            high_count = sum(1 for m in METRICS if scores.get(m, 0) >= THRESHOLD)
            if high_count >= MAJORITY:
                n_ox += 1
        accuracies[config] = n_ox / N_Q

    # Update cartesian_summary.json
    cart = json.load(open(CART_PATH))
    for r in cart:
        r["accuracy"] = round(accuracies.get(r["config"], 0.0), 4)
    json.dump(cart, open(CART_PATH, "w"), ensure_ascii=False, indent=2)
    print(f"\nUpdated {CART_PATH}")

    # Top 10 by accuracy
    cart_sorted = sorted(cart, key=lambda x: -x["accuracy"])
    print(f"\nTop 10 by accuracy (Phase 5 majority-O rule):")
    print(f"{'rank':<5}{'config':<60}{'MRR':>7}{'Hit@1':>7}{'judge':>7}{'acc':>7}")
    for i, r in enumerate(cart_sorted[:10], 1):
        print(f"{i:<5}{r['config'][:58]:<60}{r['MRR']:>7.4f}{r['Hit@1']:>7.3f}{r['judge_mean']:>7.3f}{r['accuracy']:>7.3f}")

    print(f"\nMRR-1위 config:")
    mrr1 = max(cart, key=lambda x: x["MRR"])
    print(f"  {mrr1['config']}: MRR={mrr1['MRR']:.4f} acc={mrr1['accuracy']:.3f}")
    print(f"Judge-1위 config:")
    j1 = max(cart, key=lambda x: x["judge_mean"])
    print(f"  {j1['config']}: judge={j1['judge_mean']:.3f} acc={j1['accuracy']:.3f}")
    print(f"Accuracy-1위 config:")
    a1 = max(cart, key=lambda x: x["accuracy"])
    print(f"  {a1['config']}: acc={a1['accuracy']:.3f} MRR={a1['MRR']:.4f} judge={a1['judge_mean']:.3f}")


if __name__ == "__main__":
    main()
