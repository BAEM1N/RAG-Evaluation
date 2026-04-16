#!/usr/bin/env python3
"""
allganize Auto Evaluate 재현 (병렬 버전, LLM-as-judge voting)

원본: TonicAI + MLflow ×2 + Claude3-opus → 4개 투표
우리: GPT-4o-mini로 4개 메트릭 평가 → 투표
"""
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "baseline_langchain"

client = OpenAI()
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
EVAL_WORKERS = int(os.environ.get("EVAL_WORKERS", "20"))

EVAL_PROMPTS = {
    "similarity": """Evaluate answer similarity between generated and reference.

Question: {question}
Reference: {target}
Generated: {generated}

Score 1-5 (5=essentially same meaning). Return ONLY the integer.""",

    "correctness": """Evaluate factual correctness.

Question: {question}
Reference: {target}
Generated: {generated}

Score 1-5 (5=fully correct). Return ONLY the integer.""",

    "completeness": """Evaluate completeness (covers all key points).

Question: {question}
Reference: {target}
Generated: {generated}

Score 1-5 (5=fully complete). Return ONLY the integer.""",

    "faithfulness": """Evaluate faithfulness (no hallucination).

Question: {question}
Reference: {target}
Generated: {generated}

Score 1-5 (5=fully grounded). Return ONLY the integer.""",
}


def score_single(metric: str, question: str, target: str, generated: str) -> int:
    prompt = EVAL_PROMPTS[metric].format(
        question=question, target=target, generated=generated
    )
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        for c in text:
            if c.isdigit():
                return int(c)
        return 0
    except Exception:
        return 0


def evaluate_item(item: dict, threshold: int = 4) -> dict:
    question = item["question"]
    target = item["target_answer"]
    generated = item.get("generated_answer") or ""

    if not generated:
        return {**item, "votes": {}, "o_count": 0, "result": "X"}

    votes = {}
    # 4개 메트릭 병렬
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(score_single, m, question, target, generated): m
            for m in EVAL_PROMPTS
        }
        for fut in as_completed(futures):
            m = futures[fut]
            votes[m] = fut.result()

    o_count = sum(1 for s in votes.values() if s >= threshold)
    result = "O" if o_count >= 2 else "X"

    return {**item, "votes": votes, "o_count": o_count, "result": result}


def main():
    llm_model = os.environ.get("LLM_MODEL", "gpt-4-turbo")
    result_path = RESULTS_DIR / f"baseline_{llm_model}_results.json"
    if not result_path.exists():
        print(f"ERROR: {result_path} 없음. baseline_langchain.py 먼저 실행.")
        return

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    print(f"평가: {len(results)}개 | Judge: {JUDGE_MODEL} | Workers: {EVAL_WORKERS}")

    evaluated = []
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
        futures = [ex.submit(evaluate_item, item) for item in results]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Eval"):
            evaluated.append(fut.result())

    # 도메인별 집계
    domains = {}
    for r in evaluated:
        d = r.get("domain", "unknown")
        domains.setdefault(d, {"O": 0, "X": 0})[r["result"]] += 1

    print(f"\n=== 결과 (LLM={llm_model} / Judge={JUDGE_MODEL}) ===")
    total_o = total_x = 0
    for d in sorted(domains):
        counts = domains[d]
        total = counts["O"] + counts["X"]
        acc = counts["O"] / total if total else 0
        print(f"  {d:10s}: {counts['O']:3d}/{total} ({acc:.1%})")
        total_o += counts["O"]
        total_x += counts["X"]
    total = total_o + total_x
    overall = total_o / total if total else 0
    print(f"  {'Total':10s}: {total_o:3d}/{total} ({overall:.1%})")

    # 리더보드 비교
    print(f"\n=== 리더보드 비교 ===")
    print(f"  우리 ({llm_model}):             {overall:.1%} ({total_o}/{total})")
    print(f"  원본 Langchain(gpt-4-turbo):  61.0% (183/300)")
    print(f"  원본 Langchain(gpt-3.5-turbo): 39.3% (118/300)")
    diff = overall - 0.610
    print(f"  차이: {diff:+.1%}p")

    # 저장
    out_path = RESULTS_DIR / f"auto_eval_{llm_model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "llm_model": llm_model,
            "judge_model": JUDGE_MODEL,
            "total": total,
            "o_count": total_o,
            "accuracy": round(overall, 4),
            "by_domain": {d: {
                "o": c["O"], "total": c["O"] + c["X"],
                "accuracy": round(c["O"] / (c["O"] + c["X"]), 4) if (c["O"] + c["X"]) else 0,
            } for d, c in domains.items()},
            "results": evaluated,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
