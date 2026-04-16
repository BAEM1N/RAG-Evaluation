#!/usr/bin/env python3
"""
RAGAS 컨텍스트 평가 (검색 품질)

대상: retrieval_cache/{embed_model}.json (21개 임베딩)
메트릭:
  - context_precision: 검색된 top-k 청크 중 정답과 관련 있는 비율
  - context_recall: GT 정답을 뒷받침하는 내용이 context에 있는 정도

Judge: gpt-5.4 (기본)
결과: results/ragas_retrieval/{embed_model}.json

Usage:
  python evaluate_ragas.py                       # 21 임베딩 전체
  python evaluate_ragas.py --models gemma-embed-300m,kure-v1  # 특정 모델
  python evaluate_ragas.py --judge gpt-5.4-mini  # 저비용
  python evaluate_ragas.py --sample 50           # 300개 중 샘플 50만
  python evaluate_ragas.py --dry-run             # 비용만 계산
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("ragas-eval")

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "retrieval_cache"
OUT_DIR = ROOT / "results" / "ragas_retrieval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── GPT-5.4 단가 (추정, 2026) ──
JUDGE_PRICING = {
    "gpt-5.4":       {"in": 2.50, "out": 10.00},
    "gpt-5.4-mini":  {"in": 0.25, "out":  2.00},
    "gpt-5.4-nano":  {"in": 0.10, "out":  0.50},
}


# ── RAGAS 프롬프트 (최소 의존성 구현) ──
CONTEXT_PRECISION_PROMPT = """Given a question, a ground-truth answer and retrieved contexts, determine for EACH context whether it is RELEVANT for arriving at the ground-truth answer.

Question: {question}
Ground-truth answer: {answer}

Contexts:
{contexts}

For each context, respond "1" if RELEVANT or "0" if NOT RELEVANT in the order they appear, as a JSON array of integers. Only output the JSON array.
Example: [1, 0, 1, 1, 0]"""


CONTEXT_RECALL_PROMPT = """Given a ground-truth answer and a retrieved context, break the answer into atomic statements and decide for EACH statement whether it is SUPPORTED by the retrieved context.

Ground-truth answer:
{answer}

Retrieved context:
{context}

Output a JSON array where each element is:
  {{"statement": "<atomic statement>", "supported": 1 or 0}}
Only output the JSON array, nothing else."""


def call_judge(client, model, prompt):
    """OpenAI API 호출. 실패 시 예외."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = resp.choices[0].message.content.strip()
    usage = resp.usage
    return content, usage.prompt_tokens, usage.completion_tokens


def parse_json_array(text):
    """LLM 출력에서 JSON 배열 추출."""
    # 마크다운 ```json 블록 제거
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    # 첫 [와 마지막 ] 사이만
    try:
        s = text.index("[")
        e = text.rindex("]") + 1
        return json.loads(text[s:e])
    except (ValueError, json.JSONDecodeError):
        return None


def score_context_precision(client, judge, item):
    """item의 top-5 chunk 각각에 대해 관련성 판정."""
    contexts_text = "\n\n".join(
        f"[{i+1}] {c[:1500]}"
        for i, c in enumerate(item.get("retrieved_texts", []))
    )
    prompt = CONTEXT_PRECISION_PROMPT.format(
        question=item["question"],
        answer=item["target_answer"],
        contexts=contexts_text,
    )
    content, in_tok, out_tok = call_judge(client, judge, prompt)
    verdicts = parse_json_array(content)
    if verdicts is None or not all(isinstance(v, (int, bool)) for v in verdicts):
        return None, in_tok, out_tok

    # Precision@K: 검색된 k개 중 관련=1의 비율 (rank-aware MAP 스타일)
    k = len(verdicts)
    if k == 0:
        return 0.0, in_tok, out_tok

    # RAGAS 방식: MAP (precision at each relevant position, averaged)
    relevant_so_far = 0
    precision_sum = 0.0
    for i, v in enumerate(verdicts):
        if v:
            relevant_so_far += 1
            precision_sum += relevant_so_far / (i + 1)
    score = (
        precision_sum / relevant_so_far if relevant_so_far > 0 else 0.0
    )
    return score, in_tok, out_tok


def score_context_recall(client, judge, item):
    """GT 정답의 각 atomic statement이 context로 지원되는지 판정."""
    context = "\n\n---\n\n".join(
        item.get("retrieved_texts", [])
    )[:8000]  # 너무 길면 자름

    prompt = CONTEXT_RECALL_PROMPT.format(
        answer=item["target_answer"],
        context=context,
    )
    content, in_tok, out_tok = call_judge(client, judge, prompt)
    items = parse_json_array(content)
    if items is None or not items:
        return None, in_tok, out_tok

    supported = sum(
        1 for it in items if isinstance(it, dict) and it.get("supported")
    )
    score = supported / len(items)
    return score, in_tok, out_tok


def load_cache_with_texts(cache_path: Path, prepared_chunks):
    """retrieval_cache는 이미 context text를 통합으로 가짐. 분리된 chunk가 필요하면 복원."""
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)

    # 각 item에 retrieved_texts (개별 청크 리스트) 생성
    chunk_lookup = {}
    for c in prepared_chunks:
        chunk_lookup[(c["file"], c["page"])] = c["text"]

    for item in cache:
        texts = []
        for f, p in zip(item.get("retrieved_files", []),
                        item.get("retrieved_pages", [])):
            # 같은 (file, page) 청크 매칭 — 단순 매칭
            texts.append(chunk_lookup.get((f, p), ""))
        item["retrieved_texts"] = texts

    return cache


def estimate_cost(n_items, judge):
    """비용 추정: metric 2개 × item 당 호출."""
    # 실측 기반 평균 토큰
    avg_in_per_call = 2500
    avg_out_per_call = 500
    total_calls = n_items * 2
    total_in = total_calls * avg_in_per_call / 1_000_000
    total_out = total_calls * avg_out_per_call / 1_000_000
    price = JUDGE_PRICING.get(judge, JUDGE_PRICING["gpt-5.4"])
    cost = total_in * price["in"] + total_out * price["out"]
    return {
        "calls": total_calls,
        "input_M": round(total_in, 2),
        "output_M": round(total_out, 2),
        "cost_usd": round(cost, 2),
    }


def evaluate_embedding(
    client, judge, embed_name, cache, parallel, sample
):
    """단일 임베딩에 대해 context_precision & context_recall 계산."""
    if sample and sample < len(cache):
        cache = cache[:sample]

    n = len(cache)
    logger.info(f"  {embed_name}: {n} 질문 평가")

    results = [None] * n
    total_in = total_out = 0
    t_start = time.time()

    def work(idx):
        item = cache[idx]
        prec, in1, out1 = score_context_precision(client, judge, item)
        rec, in2, out2 = score_context_recall(client, judge, item)
        return idx, {
            "question": item["question"],
            "domain": item.get("domain", ""),
            "context_type": item.get("context_type", ""),
            "context_precision": prec,
            "context_recall": rec,
        }, in1 + in2, out1 + out2

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(work, i): i for i in range(n)}
        done = 0
        for future in as_completed(futures):
            try:
                idx, r, in_tok, out_tok = future.result()
                results[idx] = r
                total_in += in_tok
                total_out += out_tok
            except Exception as e:
                logger.warning(f"    평가 실패: {e}")
            done += 1
            if done % 25 == 0:
                logger.info(
                    f"    [{done}/{n}] tok:{total_in}+{total_out}"
                )

    # 집계
    prec_scores = [r["context_precision"] for r in results if r and r["context_precision"] is not None]
    rec_scores = [r["context_recall"] for r in results if r and r["context_recall"] is not None]

    summary = {
        "embed_model": embed_name,
        "judge": judge,
        "total": n,
        "context_precision_avg": round(sum(prec_scores) / len(prec_scores), 4) if prec_scores else None,
        "context_recall_avg": round(sum(rec_scores) / len(rec_scores), 4) if rec_scores else None,
        "prec_n": len(prec_scores),
        "rec_n": len(rec_scores),
        "total_time_sec": round(time.time() - t_start, 1),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "cost_usd": round(
            (total_in / 1e6) * JUDGE_PRICING.get(judge, JUDGE_PRICING["gpt-5.4"])["in"]
            + (total_out / 1e6) * JUDGE_PRICING.get(judge, JUDGE_PRICING["gpt-5.4"])["out"],
            4,
        ),
        "results": results,
    }

    out_path = OUT_DIR / f"{embed_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"    완료: precision={summary['context_precision_avg']} "
        f"recall={summary['context_recall_avg']} "
        f"cost=${summary['cost_usd']}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="콤마로 구분된 임베딩 모델 목록 (기본: 전체)",
    )
    parser.add_argument("--judge", type=str, default="gpt-5.4")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="각 임베딩당 질문 수 제한 (0=전체)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="동시 요청 수",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="비용 추정만",
    )
    args = parser.parse_args()

    # 대상 임베딩
    cache_files = sorted(CACHE_DIR.glob("*.json"))
    if args.models:
        wanted = set(m.strip() for m in args.models.split(","))
        cache_files = [f for f in cache_files if f.stem in wanted]
    if not cache_files:
        logger.error("cache 파일 없음")
        return

    # 비용 추정
    total_items = 0
    for f in cache_files:
        with open(f) as fh:
            c = json.load(fh)
        total_items += len(c) if not args.sample else min(args.sample, len(c))
    cost_est = estimate_cost(total_items, args.judge)

    logger.info(f"=== RAGAS 컨텍스트 평가 ===")
    logger.info(f"  임베딩: {len(cache_files)}개")
    logger.info(f"  문항: {total_items}개")
    logger.info(f"  Judge: {args.judge}")
    logger.info(f"  예상 호출: {cost_est['calls']}건")
    logger.info(f"  예상 토큰: in={cost_est['input_M']}M out={cost_est['output_M']}M")
    logger.info(f"  예상 비용: ${cost_est['cost_usd']}")

    if args.dry_run:
        logger.info("(dry-run 모드, 실행 안 함)")
        return

    # OpenAI 클라이언트
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY 환경변수 필요")
        return
    client = OpenAI()

    # prepared chunks 로드 (개별 청크 텍스트 복원용)
    with open(ROOT / "data" / "prepared_chunks.json", encoding="utf-8") as f:
        prepared_chunks = json.load(f)

    # 각 임베딩 평가
    for i, cache_file in enumerate(cache_files, 1):
        embed_name = cache_file.stem
        out_path = OUT_DIR / f"{embed_name}.json"
        if out_path.exists():
            logger.info(f"\n[{i}/{len(cache_files)}] {embed_name}: 완료됨, 스킵")
            continue

        logger.info(f"\n[{i}/{len(cache_files)}] {embed_name}")
        cache = load_cache_with_texts(cache_file, prepared_chunks)
        try:
            evaluate_embedding(
                client, args.judge, embed_name, cache,
                args.parallel, args.sample,
            )
        except Exception as e:
            logger.error(f"  실패: {e}", exc_info=True)

    # 최종 요약
    print(f"\n{'='*60}")
    print("  RAGAS 컨텍스트 평가 결과")
    print(f"{'='*60}")
    print(f"{'임베딩':<28} {'Prec':>8} {'Recall':>8} {'Cost':>10}")
    print("-" * 60)
    rows = []
    for f in sorted(OUT_DIR.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        rows.append(d)
    rows.sort(key=lambda x: -(x.get("context_precision_avg") or 0))
    total_cost = 0
    for r in rows:
        p = r.get("context_precision_avg")
        rc = r.get("context_recall_avg")
        c = r.get("cost_usd", 0)
        total_cost += c
        print(f"{r['embed_model'][:28]:<28} {p if p else 0:>8.4f} {rc if rc else 0:>8.4f} ${c:>9.4f}")
    print("-" * 60)
    print(f"{'TOTAL':<28} {'':>8} {'':>8} ${total_cost:>9.2f}")


if __name__ == "__main__":
    main()
