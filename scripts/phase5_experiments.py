#!/usr/bin/env python3
"""
Phase 5: 2개 독립 실험
  실험 A (embed_compare): 1 LLM 고정 × 21 임베딩 비교
  실험 B (llm_compare):   1 임베딩 고정 × N LLM 비교

Usage:
  python phase5_experiments.py --exp A --server ai395
  python phase5_experiments.py --exp B --server ai395
  python phase5_experiments.py --exp B --server spark
"""
import sys
import json
import time
import re
import logging
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("phase5-exp")

RESULTS_DIR = Path(__file__).parent.parent / "results"
CACHE_DIR = Path(__file__).parent.parent / "data" / "retrieval_cache"

# 실험 A: 임베딩 비교용 LLM 설정들 (각 임베딩마다 이 4개 다 돌림)
FIXED_LLMS_FOR_EMBED_COMPARE = {
    "ai395": [
        {"name": "qwen3.5-27b", "think": False},
        {"name": "qwen3.5-27b", "think": True},
        {"name": "qwen3.5-35b-a3b", "think": False},
        {"name": "qwen3.5-35b-a3b", "think": True},
        {"name": "qwen3.6-35b-a3b", "think": False},
        {"name": "qwen3.6-35b-a3b", "think": True},
    ],
    "spark": [
        {"name": "qwen3.5:122b-a10b-q4_K_M", "think": False},
        {"name": "exaone3.5:32b", "think": None},
        {"name": "gpt-oss:120b", "think": None},
        {"name": "deepseek-r1:70b", "think": False},
    ],
}

# 실험 B: LLM 비교용 고정 임베딩 (MRR 1위)
FIXED_EMBED_FOR_LLM_COMPARE = "gemma-embed-300m"

# 실험 B: 서버별 LLM 목록
LLMS_FOR_COMPARE = {
    "ai395": [
        {"name": "qwen3.5-27b", "think": True},
        {"name": "qwen3.5-27b", "think": False},
        {"name": "qwen3.5-35b-a3b", "think": True},
        {"name": "qwen3.5-35b-a3b", "think": False},
    ],
    "spark": [
        {"name": "qwen3.5:122b-a10b-q4_K_M", "think": False},
        {"name": "qwen3.5:122b-a10b-q4_K_M", "think": True},
        {"name": "qwen3.5:9b-q8_0", "think": False},
        {"name": "qwen3.5:9b-q4_K_M", "think": False},
        {"name": "qwen3.5:27b-q8_0", "think": False},
        {"name": "exaone3.5:32b", "think": None},
        {"name": "gpt-oss:20b", "think": None},
        {"name": "gpt-oss:120b", "think": None},
        {"name": "phi4:14b", "think": None},
        {"name": "mistral-small:24b", "think": None},
        {"name": "deepseek-r1:70b", "think": False},
        {"name": "lfm2:24b", "think": None},
    ],
}

SERVER_URLS = {
    "ai395": "http://localhost:8080/v1",
    "spark": "http://localhost:11434",  # native API 사용
}

SERVER_PARALLEL = {
    "ai395": 4,
    "spark": 8,
}

RAG_PROMPT = (
    "다음 검색된 문맥을 사용하여 질문에 답하세요. "
    "답을 모르면 모른다고 하세요. 최대 3문장으로 간결하게 답하세요.\n\n"
    "질문: {question}\n\n문맥:\n{context}\n\n답변:"
)


def api_post(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=None) as resp:
        return json.loads(resp.read())


def call_llm(base_url, model, question, context, think_mode):
    prompt = RAG_PROMPT.format(question=question, context=context)
    messages = [{"role": "user", "content": prompt}]

    # Ollama native API (/api/chat) - think 파라미터 지원
    if base_url.endswith(":11434"):
        req = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if think_mode is False:
            req["think"] = False
            req["options"]["num_predict"] = 4096
        elif think_mode is True:
            req["think"] = True
            # no num_predict cap: let reasoning + answer run to completion
        else:
            req["options"]["num_predict"] = 4096
        data = api_post(f"{base_url}/api/chat", req)

        content = data["message"].get("content", "")
        reasoning = data["message"].get("thinking", "")
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content, reasoning, usage

    # llama.cpp OpenAI-compatible API
    req = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4096,
    }
    if think_mode is False:
        req["chat_template_kwargs"] = {"enable_thinking": False}
    elif think_mode is True:
        req["chat_template_kwargs"] = {"enable_thinking": True}

    data = api_post(f"{base_url}/chat/completions", req)

    content = data["choices"][0]["message"].get("content", "")
    reasoning = data["choices"][0]["message"].get(
        "reasoning_content", ""
    ) or data["choices"][0]["message"].get("reasoning", "")
    if "<think>" in content:
        content = re.sub(
            r"<think>.*?</think>", "", content, flags=re.DOTALL
        ).strip()

    usage = data.get("usage", {})
    return content, reasoning, usage


def result_key(exp, embed, model, think_mode):
    safe_model = model.replace(":", "_").replace("/", "_")
    suffix = (
        "_think"
        if think_mode is True
        else "_nothink" if think_mode is False else ""
    )
    return f"{exp}__{embed}__{safe_model}{suffix}"


def is_done(exp_dir, key):
    path = exp_dir / f"{key}.json"
    if not path.exists():
        return False
    with open(path) as f:
        d = json.load(f)
    return (
        sum(1 for r in d.get("results", []) if r.get("generated_answer"))
        >= 290
    )


def process_one(base_url, model_name, think_mode, item):
    try:
        t0 = time.time()
        answer, reasoning, usage = call_llm(
            base_url, model_name, item["question"], item["context"], think_mode
        )
        latency = time.time() - t0
        return {
            "question": item["question"],
            "target_answer": item["target_answer"],
            "domain": item["domain"],
            "context_type": item["context_type"],
            "generated_answer": answer,
            "reasoning_length": len(reasoning),
            "latency_sec": round(latency, 2),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "retrieved_files": item.get("retrieved_files", []),
            "retrieved_pages": item.get("retrieved_pages", []),
        }
    except Exception as e:
        return {
            "question": item["question"],
            "target_answer": item["target_answer"],
            "domain": item["domain"],
            "context_type": item["context_type"],
            "generated_answer": "",
            "error": str(e)[:200],
            "latency_sec": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def run_one_combo(
    exp,
    exp_dir,
    base_url,
    model_name,
    think_mode,
    cache_items,
    embed_name,
    parallel,
):
    key = result_key(exp, embed_name, model_name, think_mode)
    if is_done(exp_dir, key):
        logger.info(f"  {key}: 완료됨, 스킵")
        return

    think_label = (
        "think"
        if think_mode is True
        else "nothink" if think_mode is False else "default"
    )
    logger.info(f"  {key} ({think_label}, parallel={parallel})")

    results = [None] * len(cache_items)
    t_start = time.time()
    total_in = total_out = 0

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {
            ex.submit(
                process_one, base_url, model_name, think_mode, item
            ): i
            for i, item in enumerate(cache_items)
        }
        done_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            r = future.result()
            results[idx] = r
            total_in += r.get("input_tokens", 0)
            total_out += r.get("output_tokens", 0)
            done_count += 1
            if done_count % 10 == 0:
                elapsed = time.time() - t_start
                rate = done_count / elapsed * 60
                eta = (
                    (len(cache_items) - done_count) / rate
                    if rate > 0
                    else 0
                )
                ok = sum(
                    1 for r in results if r and r.get("generated_answer")
                )
                logger.info(
                    f"    [{done_count}/{len(cache_items)}] {ok}ok | "
                    f"{rate:.1f}/min | ETA {eta:.0f}min | "
                    f"tok:{total_in}+{total_out}"
                )

    total_time = time.time() - t_start
    answered = sum(1 for r in results if r and r.get("generated_answer"))

    out = {
        "experiment": exp,
        "llm": model_name,
        "think_mode": think_label,
        "embed_model": embed_name,
        "top_k": 5,
        "total": len(results),
        "answered": answered,
        "total_time_sec": round(total_time, 1),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "avg_latency_sec": round(
            sum(r.get("latency_sec", 0) for r in results if r)
            / max(len(results), 1),
            2,
        ),
        "results": results,
    }
    path = exp_dir / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(
        f"    완료: {answered}/{len(results)} | "
        f"{total_time:.0f}s | in={total_in} out={total_out}"
    )


def run_experiment_a(server):
    """실험 A: N LLM × 21 임베딩 = 21 × N 조합."""
    exp_dir = RESULTS_DIR / "phase5_exp_a_embed"
    exp_dir.mkdir(parents=True, exist_ok=True)

    base_url = SERVER_URLS[server]
    parallel = SERVER_PARALLEL[server]
    llms = FIXED_LLMS_FOR_EMBED_COMPARE[server]

    cache_files = sorted(CACHE_DIR.glob("*.json"))
    total = len(cache_files) * len(llms)
    logger.info(f"실험 A: 임베딩 비교")
    logger.info(f"  LLMs: {len(llms)}개")
    for m in llms:
        t = "think" if m["think"] is True else "nothink" if m["think"] is False else "default"
        logger.info(f"    - {m['name']} ({t})")
    logger.info(f"  임베딩: {len(cache_files)}개")
    logger.info(f"  총 조합: {total}")
    logger.info(f"  서버: {server} ({base_url}), parallel={parallel}")

    # 순서: nothink 먼저 전체, think 나중에 전체
    # (nothink는 빠르니 먼저 돌리고 결과 빨리 확보)
    def llm_priority(m):
        # nothink=0, default=1, think=2
        if m["think"] is False:
            return 0
        elif m["think"] is None:
            return 1
        else:
            return 2

    sorted_llms = sorted(llms, key=llm_priority)
    idx = 0
    for m in sorted_llms:
        t_label = "think" if m["think"] is True else "nothink" if m["think"] is False else "default"
        logger.info(f"\n=== LLM: {m['name']} ({t_label}) — 전체 임베딩 순회 ===")
        for cache_file in cache_files:
            embed_name = cache_file.stem
            with open(cache_file) as f:
                cache_items = json.load(f)
            idx += 1
            logger.info(f"\n[{idx}/{total}] {embed_name} + {m['name']} ({t_label})")
            run_one_combo(
                "expA",
                exp_dir,
                base_url,
                m["name"],
                m["think"],
                cache_items,
                embed_name,
                parallel,
            )


def run_experiment_b(server):
    """실험 B: 1 임베딩 × N LLM."""
    exp_dir = RESULTS_DIR / "phase5_exp_b_llm"
    exp_dir.mkdir(parents=True, exist_ok=True)

    base_url = SERVER_URLS[server]
    parallel = SERVER_PARALLEL[server]
    llms = LLMS_FOR_COMPARE[server]

    cache_path = CACHE_DIR / f"{FIXED_EMBED_FOR_LLM_COMPARE}.json"
    if not cache_path.exists():
        logger.error(f"캐시 없음: {cache_path}")
        return

    with open(cache_path) as f:
        cache_items = json.load(f)

    logger.info(f"실험 B: LLM 비교")
    logger.info(f"  고정 임베딩: {FIXED_EMBED_FOR_LLM_COMPARE}")
    logger.info(f"  LLM: {len(llms)}개")
    logger.info(f"  서버: {server} ({base_url}), parallel={parallel}")

    def _prio(m):
        return 0 if m["think"] is False else (1 if m["think"] is None else 2)
    llms = sorted(llms, key=_prio)

    for i, m in enumerate(llms, 1):
        think_label = (
            "think"
            if m["think"] is True
            else "nothink" if m["think"] is False else "default"
        )
        logger.info(
            f"\n[{i}/{len(llms)}] {m['name']} ({think_label})"
        )
        run_one_combo(
            "expB",
            exp_dir,
            base_url,
            m["name"],
            m["think"],
            cache_items,
            FIXED_EMBED_FOR_LLM_COMPARE,
            parallel,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=["A", "B"])
    parser.add_argument(
        "--server", required=True, choices=["ai395", "spark"]
    )
    args = parser.parse_args()

    if args.exp == "A":
        run_experiment_a(args.server)
    else:
        run_experiment_b(args.server)


if __name__ == "__main__":
    main()
