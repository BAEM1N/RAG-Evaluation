#!/usr/bin/env python3
"""LLM-as-judge aligned with allganize RAG-Evaluation-Dataset-KO methodology.

4 metric LLM calls per item → vote → O/X.
Prompts ported from MLflow answer_similarity/v1 + answer_correctness/v1
(the exact metrics allganize uses in their leaderboard).
All 4 metrics use 1-5 scale with explicit rubric.

Final: "O" when ≥2 metrics score ≥ THRESHOLD (default 4) — matches allganize.

Usage:
  JUDGE_URL=http://localhost:8080/v1 JUDGE_MODEL=gpt-oss-120b \
  JUDGE_MODE=nothink PARALLEL=4 python3 llm_judge.py <result-file>
"""
import sys, os, json, time, re, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

JUDGE_URL = os.environ.get('JUDGE_URL', 'http://localhost:8080/v1')
JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'gpt-oss-120b')
JUDGE_MODE = os.environ.get('JUDGE_MODE', 'nothink')  # nothink|default|think
PARALLEL = int(os.environ.get('PARALLEL', '4'))
THRESHOLD = int(os.environ.get('THRESHOLD', '4'))

BASE = Path(os.path.expanduser('~/RAG-Evaluation'))
IN_DIR = BASE / 'results' / 'phase5_exp_b_llm'
OUT_DIR = BASE / 'results' / 'phase5_judge'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MLflow-style explicit 1-5 rubrics (allganize leaderboard uses MLflow answer_similarity/v1 and answer_correctness/v1)
EVAL_PROMPTS = {
    "similarity": """You are an impartial judge rating answer similarity in Korean RAG.

Question: {question}
Reference (ground truth): {target}
Candidate answer: {generated}

Rate semantic similarity of the candidate to the reference on a 1-5 scale:
1 — Little to no semantic similarity; candidate does not convey the reference meaning.
2 — Partial similarity on some aspects, but major meaning diverges.
3 — Moderate similarity; core idea matches but notable differences.
4 — Candidate aligns with reference in most aspects with substantial similarity.
5 — Candidate closely aligns with the reference in all significant aspects.

Respond with ONLY the integer 1, 2, 3, 4, or 5.""",

    "correctness": """You are an impartial judge rating answer correctness in Korean RAG.

Question: {question}
Reference (ground truth): {target}
Candidate answer: {generated}

Rate factual correctness of the candidate against the reference on a 1-5 scale:
1 — Completely incorrect or contradicts the reference.
2 — Partial correctness but significant discrepancies (wrong numbers, wrong entities).
3 — Addresses several aspects accurately with some minor omissions or inaccuracies.
4 — Mostly correct with only minor omissions or inaccuracies.
5 — Correct with high accuracy and full alignment to the reference facts.

Respond with ONLY the integer 1, 2, 3, 4, or 5.""",

    "completeness": """You are an impartial judge rating answer completeness in Korean RAG.

Question: {question}
Reference (ground truth): {target}
Candidate answer: {generated}

Rate how completely the candidate covers the key points from the reference on a 1-5 scale:
1 — Very incomplete; most key points from the reference are missing.
2 — Only a small fraction of key points covered; major gaps remain.
3 — About half of the key points covered; notable omissions.
4 — Most key points covered; only minor details missing.
5 — Fully complete; every important element from the reference is present.

Respond with ONLY the integer 1, 2, 3, 4, or 5.""",

    "faithfulness": """You are an impartial judge rating answer faithfulness in Korean RAG.

Question: {question}
Reference (ground truth): {target}
Candidate answer: {generated}

Rate how faithful the candidate is — i.e., free from hallucinations or claims that contradict the reference — on a 1-5 scale:
1 — Many hallucinations or claims that contradict the reference.
2 — Several unsupported or contradictory claims alongside some valid content.
3 — Mixed; some unsupported claims but core content aligns with the reference.
4 — Mostly faithful; only one or two minor unsupported details.
5 — Fully faithful; every claim is supported by or consistent with the reference.

Respond with ONLY the integer 1, 2, 3, 4, or 5.""",
}


def _post(body: bytes, url: str):
    r = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=None) as resp:
        return json.loads(resp.read())


def call_judge_once(metric: str, question: str, target: str, generated: str) -> int:
    prompt = EVAL_PROMPTS[metric].format(question=question, target=target, generated=generated)
    messages = [{"role": "user", "content": prompt}]

    # Do not cap tokens — let the model run to natural completion within its context.
    if JUDGE_URL.endswith(':11434') or JUDGE_URL.endswith(':11434/'):
        url = JUDGE_URL.rstrip('/') + '/api/chat'
        # ollama: num_predict=-1 => unlimited
        req = {"model": JUDGE_MODEL, "messages": messages, "stream": False,
               "options": {"temperature": 0, "num_predict": -1}}
        if JUDGE_MODE == 'nothink':
            req["think"] = False
        elif JUDGE_MODE == 'think':
            req["think"] = True
        try:
            data = _post(json.dumps(req).encode(), url)
            msg = data.get("message", {})
            content = msg.get("content", "") or ""
            reasoning = msg.get("thinking", "") or msg.get("reasoning", "") or ""
        except Exception:
            return 0
    else:
        url = JUDGE_URL.rstrip('/') + '/chat/completions'
        # llama.cpp OpenAI-compat: omit max_tokens => server default (effectively unlimited within ctx)
        req = {"model": JUDGE_MODEL, "messages": messages, "temperature": 0}
        if JUDGE_MODE == 'nothink':
            req["chat_template_kwargs"] = {"enable_thinking": False}
        elif JUDGE_MODE == 'think':
            req["chat_template_kwargs"] = {"enable_thinking": True}
        try:
            data = _post(json.dumps(req).encode(), url)
            msg = data["choices"][0]["message"]
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or msg.get("reasoning", "") or ""
        except Exception:
            return 0

    # Score extraction: try content first (preferred = final answer),
    # then <think>-stripped version, then reasoning field.
    def last_score(s: str):
        """Return last digit 1-5 in the string (so 'Score: 4' or '... ends with 4' both work)."""
        if not s:
            return 0
        # Prefer patterns like 'Score: 4', '4/5', just-digit, etc.
        m = re.findall(r'\b([1-5])\b', s)
        if m:
            return int(m[-1])
        # Fallback: any digit 1-5 anywhere
        for c in reversed(s):
            if c.isdigit():
                n = int(c)
                if 1 <= n <= 5:
                    return n
        return 0

    # 1) try content as-is
    n = last_score(content)
    if n:
        return n
    # 2) try content with <think> stripped
    stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    n = last_score(stripped)
    if n:
        return n
    # 3) reasoning-only fallback
    n = last_score(reasoning)
    if n:
        return n
    return 0


def judge_item(idx: int, result_item: dict) -> tuple:
    generated = result_item.get('generated_answer') or ''
    if not generated:
        return idx, {"votes": {m: 1 for m in EVAL_PROMPTS}, "o_count": 0,
                     "result": "X", "note": "candidate empty", "latency": 0}

    q = result_item['question']
    tgt = result_item['target_answer']
    votes = {}
    t0 = time.time()
    for m in EVAL_PROMPTS:
        votes[m] = call_judge_once(m, q, tgt, generated)

    o_count = sum(1 for s in votes.values() if s >= THRESHOLD)
    result = "O" if o_count >= 2 else "X"
    return idx, {"votes": votes, "o_count": o_count, "result": result,
                 "latency": round(time.time() - t0, 1)}


def main():
    if len(sys.argv) < 2:
        print("Usage: llm_judge.py <exp_b_result_file>")
        sys.exit(1)
    in_path = IN_DIR / sys.argv[1]
    if not in_path.exists():
        print(f"Not found: {in_path}"); sys.exit(1)

    judge_tag = JUDGE_MODEL.replace(':', '_').replace('/', '_') + f'_{JUDGE_MODE}'
    out_path = OUT_DIR / f'judge_{judge_tag}__{sys.argv[1]}'

    data = json.load(open(in_path))
    results = data['results']

    scores = [None] * len(results)
    if out_path.exists():
        try:
            prev = json.load(open(out_path))
            for i, s in enumerate(prev.get('scores', [])):
                if s and s.get('result') in ('O', 'X'):
                    scores[i] = s
        except Exception:
            pass

    todo = [i for i, s in enumerate(scores) if s is None]
    print(f'Judge: {JUDGE_MODEL} ({JUDGE_MODE}), threshold={THRESHOLD}')
    print(f'Target: {sys.argv[1]}')
    print(f'Total: {len(results)}, pending: {len(todo)}, parallel: {PARALLEL}')
    if not todo:
        print('Nothing to judge.')
        return

    save_lock = threading.Lock()
    done = 0
    t_start = time.time()

    def _save():
        with save_lock:
            o = sum(1 for s in scores if s and s.get('result') == 'O')
            x = sum(1 for s in scores if s and s.get('result') == 'X')
            scored = o + x
            out = {
                'judge_model': JUDGE_MODEL, 'judge_mode': JUDGE_MODE,
                'threshold': THRESHOLD, 'metrics': list(EVAL_PROMPTS.keys()),
                'original_file': sys.argv[1], 'llm': data.get('llm'),
                'total': len(results), 'scored': scored,
                'o_count': o, 'x_count': x,
                'accuracy': round(o / scored, 4) if scored else None,
                'scores': scores,
            }
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_item, i, results[i]) for i in todo]
        for fut in as_completed(futs):
            idx, sc = fut.result()
            with save_lock:
                scores[idx] = sc
                done += 1
            if done % 10 == 0 or done == len(todo):
                _save()
                elapsed = time.time() - t_start
                rate = done / elapsed * 60
                eta = (len(todo) - done) / rate if rate > 0 else 0
                o = sum(1 for s in scores if s and s.get('result') == 'O')
                x = sum(1 for s in scores if s and s.get('result') == 'X')
                print(f'  [{done}/{len(todo)}] O={o} X={x} | {rate:.1f}/min | ETA {eta:.0f}min')

    _save()

    o = sum(1 for s in scores if s and s.get('result') == 'O')
    x = sum(1 for s in scores if s and s.get('result') == 'X')
    total = o + x
    if total:
        print(f'\nFinal: {o}/{total} = {o/total*100:.2f}% accuracy')
        for m in EVAL_PROMPTS:
            vals = [s['votes'].get(m, 0) for s in scores if s and s.get('votes', {}).get(m)]
            if vals:
                print(f'  {m:14} avg={sum(vals)/len(vals):.2f}')


if __name__ == '__main__':
    main()
