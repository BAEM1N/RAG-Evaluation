#!/usr/bin/env python3
"""
Phase 5 실험 B (프로바이더): OpenRouter + Friendli.ai

고정 임베딩: gemma-embed-300m (MRR 1위, retrieval_cache 사용)
LLM: OpenRouter 22개 + Friendli.ai 고유 5개 = ~27개

Usage:
  python phase5_provider.py --provider openrouter
  python phase5_provider.py --provider friendli
  python phase5_provider.py --provider all
  python phase5_provider.py --models openai/gpt-5.4-mini,anthropic/claude-sonnet-4.6
  python phase5_provider.py --dry-run
  python phase5_provider.py --sample 30   # 샘플 테스트
"""
import os
import sys
import json
import time
import re
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("phase5-provider")

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "retrieval_cache"
OUT_DIR = ROOT / "results" / "phase5_exp_b_provider"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_EMBED = "gemma-embed-300m"

RAG_PROMPT = (
    "다음 검색된 문맥을 사용하여 질문에 답하세요. "
    "답을 모르면 모른다고 하세요. 최대 3문장으로 간결하게 답하세요.\n\n"
    "질문: {question}\n\n문맥:\n{context}\n\n답변:"
)

# ── OpenRouter 모델 (22개) ──
OPENROUTER_MODELS = [
    # Premium
    {"id": "openai/gpt-5.4-pro", "think": True, "tier": "premium"},
    {"id": "openai/gpt-5.4", "think": True, "tier": "premium"},
    {"id": "openai/gpt-5.4-mini", "think": True, "tier": "budget"},
    {"id": "openai/gpt-5.4-nano", "think": True, "tier": "budget"},
    {"id": "openai/gpt-5.3-chat", "think": False, "tier": "standard"},
    {"id": "anthropic/claude-opus-4.6", "think": True, "tier": "premium"},
    {"id": "anthropic/claude-sonnet-4.6", "think": True, "tier": "standard"},
    {"id": "google/gemini-3.1-pro-preview", "think": True, "tier": "premium"},
    {"id": "google/gemini-3.1-flash-lite-preview", "think": False, "tier": "budget"},
    # Global
    {"id": "x-ai/grok-4.20", "think": True, "tier": "standard"},
    {"id": "cohere/command-a", "think": False, "tier": "standard"},
    {"id": "perplexity/sonar-reasoning-pro", "think": True, "tier": "standard"},
    # Asia
    {"id": "qwen/qwen3.6-plus", "think": False, "tier": "standard"},
    {"id": "qwen/qwen3.5-flash", "think": False, "tier": "budget"},
    {"id": "qwen/qwen3-max-thinking", "think": True, "tier": "standard"},
    {"id": "z-ai/glm-5.1", "think": True, "tier": "standard"},
    {"id": "z-ai/glm-4.7", "think": False, "tier": "budget"},
    {"id": "minimax/minimax-m2.7", "think": False, "tier": "budget"},
    {"id": "moonshotai/kimi-k2.5", "think": True, "tier": "standard"},
    # Open
    {"id": "nex-agi/deepseek-v3.1-nex-n1", "think": False, "tier": "budget"},
    {"id": "mistralai/mistral-small-2603", "think": False, "tier": "budget"},
    {"id": "nvidia/nemotron-3-super-120b-a12b", "think": False, "tier": "budget"},
    # Korean
    {"id": "upstage/solar-pro-3", "think": False, "tier": "budget"},
]

# ── Friendli.ai 고유 모델 (5개, OpenRouter에 없는 것) ──
FRIENDLI_MODELS = [
    {"id": "LGAI-EXAONE/K-EXAONE-236B-A23B", "think": False, "tier": "standard"},
    {"id": "Qwen/Qwen3-235B-A22B-Instruct-2507", "think": False, "tier": "standard"},
    {"id": "deepseek-ai/DeepSeek-V3.2", "think": False, "tier": "standard"},
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "think": False, "tier": "standard"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "think": False, "tier": "budget"},
]


def safe_model_name(model_id):
    return model_id.replace("/", "__").replace(":", "_")


def is_done(key):
    path = OUT_DIR / f"{key}.json"
    if not path.exists():
        return False
    with open(path) as f:
        d = json.load(f)
    return sum(1 for r in d.get("results", []) if r.get("generated_answer")) >= 290


def make_client(provider, model_config, think_mode):
    """LangChain ChatModel 생성."""
    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        kwargs = {
            "model": model_config["id"],
            "temperature": 0,
            "max_tokens": 4096,
            "max_retries": 3,
        }
        if think_mode and model_config.get("think"):
            kwargs["reasoning"] = {"effort": "medium"}
        return ChatOpenRouter(**kwargs)

    elif provider == "friendli":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_config["id"],
            base_url="https://api.friendli.ai/serverless/v1",
            api_key=os.environ.get("FRIENDLI_TOKEN", ""),
            temperature=0,
            max_tokens=4096,
            max_retries=3,
        )


def run_model(provider, model_config, cache_items, parallel=20, sample=0):
    model_id = model_config["id"]
    think_mode = model_config.get("think", False)
    key = f"{provider}__{safe_model_name(model_id)}"

    if is_done(key):
        logger.info(f"  {key}: 완료됨, 스킵")
        return

    client = make_client(provider, model_config, think_mode)
    items = cache_items[:sample] if sample else cache_items

    logger.info(f"  {key} ({len(items)} items, parallel={parallel})")

    def process(item):
        prompt = RAG_PROMPT.format(question=item["question"], context=item["context"])
        try:
            t0 = time.time()
            resp = client.invoke(prompt)
            latency = time.time() - t0

            content = resp.content if hasattr(resp, "content") else str(resp)
            if isinstance(content, list):
                content = "".join(
                    b["text"] if isinstance(b, dict) and b.get("type") == "text"
                    else str(b)
                    for b in content
                )
            if "<think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            usage = getattr(resp, "usage_metadata", None) or {}
            return {
                "question": item["question"],
                "target_answer": item["target_answer"],
                "domain": item.get("domain", ""),
                "context_type": item.get("context_type", ""),
                "generated_answer": content,
                "latency_sec": round(latency, 2),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "retrieved_files": item.get("retrieved_files", []),
                "retrieved_pages": item.get("retrieved_pages", []),
            }
        except Exception as e:
            return {
                "question": item["question"],
                "target_answer": item["target_answer"],
                "domain": item.get("domain", ""),
                "context_type": item.get("context_type", ""),
                "generated_answer": "",
                "error": str(e)[:200],
                "latency_sec": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

    results = [None] * len(items)
    t_start = time.time()
    total_in = total_out = 0

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(process, item): i for i, item in enumerate(items)}
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
                eta = (len(items) - done_count) / rate if rate > 0 else 0
                ok = sum(1 for r in results if r and r.get("generated_answer"))
                logger.info(
                    f"    [{done_count}/{len(items)}] {ok}ok | "
                    f"{rate:.1f}/min | ETA {eta:.0f}min | "
                    f"tok:{total_in}+{total_out}"
                )

    total_time = time.time() - t_start
    answered = sum(1 for r in results if r and r.get("generated_answer"))

    out = {
        "provider": provider,
        "llm": model_id,
        "tier": model_config.get("tier", ""),
        "think_mode": "think" if think_mode else "default",
        "embed_model": FIXED_EMBED,
        "top_k": 5,
        "total": len(results),
        "answered": answered,
        "total_time_sec": round(total_time, 1),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "avg_latency_sec": round(
            sum(r.get("latency_sec", 0) for r in results if r) / max(len(results), 1), 2
        ),
        "results": results,
    }
    path = OUT_DIR / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(
        f"    완료: {answered}/{len(results)} | "
        f"{total_time:.0f}s | in={total_in} out={total_out}"
    )


def estimate_cost(models, n_items):
    """대략적인 비용 추정 (조 당 1k in + 300 out)."""
    price_map = {
        # OpenRouter (per 1M token)
        "openai/gpt-5.4-pro": (30.0, 180.0),
        "openai/gpt-5.4": (2.5, 15.0),
        "openai/gpt-5.4-mini": (0.75, 4.5),
        "openai/gpt-5.4-nano": (0.2, 1.25),
        "openai/gpt-5.3-chat": (1.75, 14.0),
        "anthropic/claude-opus-4.6": (5.0, 25.0),
        "anthropic/claude-sonnet-4.6": (3.0, 15.0),
        "google/gemini-3.1-pro-preview": (2.0, 12.0),
        "google/gemini-3.1-flash-lite-preview": (0.25, 1.5),
        "x-ai/grok-4.20": (2.0, 6.0),
        "cohere/command-a": (2.0, 10.0),
        "perplexity/sonar-reasoning-pro": (3.0, 15.0),
        "qwen/qwen3.6-plus": (0.33, 1.95),
        "qwen/qwen3.5-flash": (0.065, 0.26),
        "qwen/qwen3-max-thinking": (0.78, 3.9),
        "z-ai/glm-5.1": (0.95, 3.15),
        "z-ai/glm-4.7": (0.39, 1.75),
        "minimax/minimax-m2.7": (0.3, 1.2),
        "moonshotai/kimi-k2.5": (0.38, 1.72),
        "nex-agi/deepseek-v3.1-nex-n1": (0.14, 0.5),
        "mistralai/mistral-small-2603": (0.15, 0.6),
        "nvidia/nemotron-3-super-120b-a12b": (0.1, 0.5),
        "upstage/solar-pro-3": (0.15, 0.6),
        # Friendli
        "LGAI-EXAONE/K-EXAONE-236B-A23B": (0.2, 0.8),
        "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.2, 0.8),
        "deepseek-ai/DeepSeek-V3.2": (0.5, 1.5),
        "meta-llama/Llama-3.3-70B-Instruct": (0.6, 0.6),
        "meta-llama/Llama-3.1-8B-Instruct": (0.1, 0.1),
    }
    total = 0.0
    rows = []
    for m in models:
        p_in, p_out = price_map.get(m["id"], (1.0, 5.0))
        in_toks = n_items * 1000 / 1e6
        out_toks = n_items * 300 / 1e6
        cost = in_toks * p_in + out_toks * p_out
        total += cost
        rows.append((m["id"], cost))
    return total, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["openrouter", "friendli", "all"],
        default="all",
    )
    parser.add_argument(
        "--models",
        default="",
        help="콤마 구분 모델 ID 목록 (지정 시 이것만 실행)",
    )
    parser.add_argument("--sample", type=int, default=0, help="샘플링 개수")
    parser.add_argument("--parallel", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 캐시 로드
    cache_path = CACHE_DIR / f"{FIXED_EMBED}.json"
    if not cache_path.exists():
        logger.error(f"캐시 없음: {cache_path}")
        return
    with open(cache_path, encoding="utf-8") as f:
        cache_items = json.load(f)
    logger.info(f"고정 임베딩: {FIXED_EMBED} ({len(cache_items)} 항목)")

    # 모델 목록 구성
    tasks = []  # (provider, model_config)
    if args.provider in ("openrouter", "all"):
        for m in OPENROUTER_MODELS:
            tasks.append(("openrouter", m))
    if args.provider in ("friendli", "all"):
        for m in FRIENDLI_MODELS:
            tasks.append(("friendli", m))

    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        tasks = [(p, m) for p, m in tasks if m["id"] in wanted]

    n_items = args.sample if args.sample else len(cache_items)
    logger.info(f"대상 모델: {len(tasks)}개, 문항: {n_items}개")

    # 비용 추정
    total_cost, rows = estimate_cost([m for _, m in tasks], n_items)
    logger.info(f"예상 비용: ${total_cost:.2f}")
    if args.dry_run:
        for mid, c in sorted(rows, key=lambda x: -x[1]):
            logger.info(f"  {mid}: ${c:.4f}")
        logger.info("(dry-run, 실제 호출 없음)")
        return

    # 환경변수 체크
    if args.provider in ("openrouter", "all") and not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY 환경변수 필요")
        return
    if args.provider in ("friendli", "all") and not os.environ.get("FRIENDLI_TOKEN"):
        logger.error("FRIENDLI_TOKEN 환경변수 필요")
        return

    # 실행
    for i, (provider, m) in enumerate(tasks, 1):
        logger.info(f"\n[{i}/{len(tasks)}] {provider} / {m['id']}")
        try:
            run_model(provider, m, cache_items, args.parallel, args.sample)
        except Exception as e:
            logger.error(f"  모델 실패: {e}", exc_info=True)

    # 요약
    print(f"\n{'='*70}")
    print(f"  Phase 5 실험 B (프로바이더) 완료")
    print(f"{'='*70}")
    print(f"{'Provider':<12} {'Model':<50} {'Answered':>10} {'Cost':>8}")
    print("-" * 70)
    for f in sorted(OUT_DIR.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        cost_in = d.get("total_input_tokens", 0) / 1e6
        cost_out = d.get("total_output_tokens", 0) / 1e6
        # 대략 추정
        print(
            f"{d.get('provider',''):<12} {d.get('llm',''):<50} "
            f"{d.get('answered',0)}/{d.get('total',0):<5}"
        )


if __name__ == "__main__":
    main()
