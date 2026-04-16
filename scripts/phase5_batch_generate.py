#!/usr/bin/env python3
"""
Phase 5: 배치 병렬 LLM 생성
- 사전 계산된 retrieval cache 사용
- ThreadPoolExecutor로 동시 N개 요청
- reasoning ON/OFF 자동 처리
- 서버 URL / 모델 목록을 인자로 받음

Usage:
  python phase5_batch_generate.py --server ai395
  python phase5_batch_generate.py --server spark
"""
import sys
import json
import time
import re
import logging
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("phase5-batch")

RESULTS_DIR = Path(__file__).parent.parent / "results"
CACHE_DIR = Path(__file__).parent.parent / "data" / "retrieval_cache"
P5_DIR = RESULTS_DIR / "phase5_llm"
P5_DIR.mkdir(parents=True, exist_ok=True)

RAG_PROMPT = (
    "다음 검색된 문맥을 사용하여 질문에 답하세요. "
    "답을 모르면 모른다고 하세요. 최대 3문장으로 간결하게 답하세요.\n\n"
    "질문: {question}\n\n문맥:\n{context}\n\n답변:"
)

SERVERS = {
    "ai395": {
        "url": "http://localhost:8080/v1",
        "models": [
            {"name": "qwen3.5-27b", "think": True},
            {"name": "qwen3.5-27b", "think": False},
            {"name": "qwen3.5-35b-a3b", "think": True},
            {"name": "qwen3.5-35b-a3b", "think": False},
        ],
        "parallel": 4,
    },
    "spark": {
        "url": "http://localhost:11434/v1",
        "models": [
            # Qwen (양자화 비교)
            {"name": "qwen3.5:122b-a10b-q4_K_M", "think": True},
            {"name": "qwen3.5:122b-a10b-q4_K_M", "think": False},
            {"name": "qwen3.5:9b-q8_0", "think": True},
            {"name": "qwen3.5:9b-q4_K_M", "think": True},
            {"name": "qwen3.5:27b-q8_0", "think": True},
            # 한국 모델
            {"name": "exaone3.5:32b", "think": None},
            {"name": "ingu627/exaone4.0:32b", "think": None},
            # 글로벌 최신
            {"name": "gemma4:27b", "think": None},
            {"name": "gpt-oss:20b", "think": None},
            {"name": "gpt-oss:120b", "think": None},
            {"name": "nemotron-3-super:120b-a12b-q4_K_M", "think": None},
            {"name": "gemma3:27b", "think": None},
            {"name": "llama4-scout:17b-16e-instruct", "think": None},
            {"name": "phi4:14b", "think": None},
            {"name": "mistral-small:24b", "think": None},
            {"name": "deepseek-r1:70b", "think": True},
            {"name": "deepseek-r1:70b", "think": False},
            {"name": "lfm2:24b", "think": None},
            {"name": "glm-5.1", "think": None},
            {"name": "solar:10.7b", "think": None},
        ],
        "parallel": 8,
    },
}


def api_post(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=None) as resp:
        return json.loads(resp.read())


def call_llm(base_url, model, question, context, think_mode):
    messages = []
    if think_mode is False:
        messages.append({"role": "system", "content": "/no_think"})
    prompt = RAG_PROMPT.format(question=question, context=context)
    messages.append({"role": "user", "content": prompt})

    data = api_post(
        f"{base_url}/chat/completions",
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 4096,
        },
    )

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


def result_key(embed, model, think_mode):
    safe_model = model.replace(":", "_").replace("/", "_")
    suffix = (
        "_think"
        if think_mode is True
        else "_nothink" if think_mode is False else ""
    )
    return f"{embed}__{safe_model}{suffix}"


def is_done(key):
    path = P5_DIR / f"{key}.json"
    if not path.exists():
        return False
    with open(path) as f:
        d = json.load(f)
    return (
        sum(1 for r in d.get("results", []) if r.get("generated_answer"))
        >= 290
    )


def process_one(base_url, model_name, think_mode, item):
    """단일 질문 처리."""
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
            "error": str(e),
            "latency_sec": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def run_generation(
    base_url, model_name, think_mode, cache_items, embed_name, parallel
):
    key = result_key(embed_name, model_name, think_mode)
    if is_done(key):
        logger.info(f"  {key}: 완료됨, 스킵")
        return None

    think_label = (
        "think"
        if think_mode is True
        else "nothink" if think_mode is False else "default"
    )
    logger.info(f"  {key} ({think_label}, parallel={parallel})")

    results = []
    t_start = time.time()
    total_in = total_out = 0

    if parallel <= 1:
        for i, item in enumerate(cache_items):
            r = process_one(base_url, model_name, think_mode, item)
            results.append(r)
            total_in += r.get("input_tokens", 0)
            total_out += r.get("output_tokens", 0)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed * 60
                eta = (len(cache_items) - i - 1) / rate if rate > 0 else 0
                ok = sum(1 for r in results if r.get("generated_answer"))
                logger.info(
                    f"    [{i+1}/{len(cache_items)}] {ok}ok | "
                    f"{rate:.1f}/min | ETA {eta:.0f}min | "
                    f"tok:{total_in}+{total_out}"
                )
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            future_map = {
                ex.submit(
                    process_one, base_url, model_name, think_mode, item
                ): i
                for i, item in enumerate(cache_items)
            }
            result_buf = [None] * len(cache_items)
            done_count = 0
            for future in as_completed(future_map):
                idx = future_map[future]
                r = future.result()
                result_buf[idx] = r
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
                        1
                        for r in result_buf
                        if r and r.get("generated_answer")
                    )
                    logger.info(
                        f"    [{done_count}/{len(cache_items)}] {ok}ok | "
                        f"{rate:.1f}/min | ETA {eta:.0f}min | "
                        f"tok:{total_in}+{total_out}"
                    )
            results = result_buf

    total_time = time.time() - t_start
    answered = sum(1 for r in results if r and r.get("generated_answer"))

    out = {
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
    path = P5_DIR / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(
        f"    완료: {answered}/{len(results)} | "
        f"{total_time:.0f}s | in={total_in} out={total_out}"
    )
    return key


def check_model(base_url, model):
    try:
        data = api_post(
            f"{base_url}/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10,
            },
        )
        return bool(data.get("choices"))
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server", required=True, choices=["ai395", "spark"]
    )
    args = parser.parse_args()

    server = SERVERS[args.server]
    base_url = server["url"]
    parallel = server["parallel"]

    cache_files = sorted(CACHE_DIR.glob("*.json"))
    if not cache_files:
        logger.error(
            "검색 캐시 없음. precompute_retrieval.py 먼저 실행"
        )
        return

    logger.info(f"서버: {args.server} ({base_url})")
    logger.info(f"임베딩 캐시: {len(cache_files)}개")
    logger.info(f"LLM 모델: {len(server['models'])}개")
    logger.info(f"병렬: {parallel}")

    available_models = []
    checked = set()
    for m in server["models"]:
        if m["name"] not in checked:
            logger.info(f"  모델 확인: {m['name']}...")
            if check_model(base_url, m["name"]):
                logger.info("    -> OK")
                checked.add(m["name"])
            else:
                logger.info("    -> 사용 불가, 스킵")
                continue
        available_models.append(m)

    logger.info(f"\n사용 가능: {len(available_models)}개 조합")

    total_combos = len(cache_files) * len(available_models)
    done = 0

    for cache_file in cache_files:
        embed_name = cache_file.stem
        with open(cache_file) as f:
            cache_items = json.load(f)

        for m in available_models:
            done += 1
            think_label = (
                "think"
                if m["think"] is True
                else "nothink" if m["think"] is False else "default"
            )
            logger.info(
                f"\n[{done}/{total_combos}] "
                f"{embed_name} + {m['name']} ({think_label})"
            )
            run_generation(
                base_url,
                m["name"],
                m["think"],
                cache_items,
                embed_name,
                parallel,
            )

    logger.info("\n=== 전체 완료 ===")
    logger.info(
        f"결과 파일: {len(list(P5_DIR.glob('*.json')))}개"
    )


if __name__ == "__main__":
    main()
