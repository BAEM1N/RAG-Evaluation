#!/usr/bin/env python3
"""
Phase 5: LLM 생성 비교 (AI-395 원격 실행)

Phase 4 최적 임베딩으로 검색 → 로컬 LLM으로 답변 생성 → 로컬 LLM-as-judge 평가.
100% 로컬, OpenAI API 불필요.
"""
import sys
import json
import time
import logging
import httpx
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("phase5")

GATEWAY_URL = "http://localhost:8080/v1"
GATEWAY_BASE = "http://localhost:8080"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Phase 4 결과에서 최적 임베딩 자동 선택
def find_best_embedding() -> str:
    """Phase 4 결과에서 MRR 최고 모델 반환."""
    p4_dir = RESULTS_DIR / "phase4_embedding"
    best_model = None
    best_mrr = 0
    for f in p4_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        mrr = data.get("metrics", {}).get("mrr", 0)
        if mrr > best_mrr:
            best_mrr = mrr
            best_model = f.stem
    logger.info(f"최적 임베딩: {best_model} (MRR={best_mrr:.4f})")
    return best_model


LLM_MODELS = {
    "qwen3.5-27b":     {"vram": 26.0},
    "qwen3.5-35b-a3b": {"vram": 20.0},
}

# 평가할 LLM을 judge로도 사용 (교차 평가)
JUDGE_MODEL = "qwen3.5-27b"

RAG_PROMPT = (
    "다음 검색된 문맥을 사용하여 질문에 답하세요. "
    "답을 모르면 모른다고 하세요. 최대 3문장으로 간결하게 답하세요.\n\n"
    "질문: {question}\n\n문맥:\n{context}\n\n답변:"
)

JUDGE_PROMPT = (
    "당신은 RAG 시스템의 답변 품질을 평가하는 심사관입니다.\n\n"
    "질문: {question}\n"
    "정답: {reference}\n"
    "생성된 답변: {generated}\n\n"
    "위 생성된 답변이 정답과 비교하여 얼마나 정확하고 유사한지 1~5점으로 평가하세요.\n"
    "5: 정답과 동일한 의미, 4: 대부분 정확, 3: 부분적으로 정확, 2: 약간만 관련, 1: 완전히 틀림\n\n"
    "점수만 숫자로 답하세요:"
)


def ensure_models_loaded(embed_model: str):
    """필요한 모델들 로드 확인."""
    with httpx.Client(timeout=120) as client:
        resp = client.get(f"{GATEWAY_BASE}/v1/models")
        data = resp.json()
        running = {m["alias"] for m in data["running"]}
        running_embeds = {m["alias"] for m in data["running"] if m["mode"] == "embedding"}

        # 불필요한 임베딩 모델 언로드
        for alias in running_embeds:
            if alias != embed_model:
                client.post(f"{GATEWAY_BASE}/v1/models/unload", json={"model": alias})
                logger.info(f"  언로드: {alias}")
                time.sleep(2)

        # 임베딩 모델 로드
        if embed_model not in running:
            client.post(f"{GATEWAY_BASE}/v1/models/load", json={"model": embed_model})
            logger.info(f"  로드: {embed_model}")
            time.sleep(5)

        # LLM 모델 확인
        for llm in LLM_MODELS:
            if llm not in running:
                client.post(f"{GATEWAY_BASE}/v1/models/load", json={"model": llm})
                logger.info(f"  로드: {llm}")
                time.sleep(5)

        # 헬스체크
        for _ in range(30):
            r = client.get(f"{GATEWAY_BASE}/v1/models")
            alive = {m["alias"] for m in r.json()["running"] if m["alive"]}
            needed = {embed_model} | set(LLM_MODELS.keys())
            if needed.issubset(alive):
                logger.info(f"모델 준비 완료: {needed}")
                return True
            time.sleep(2)

    return False


def call_llm(model: str, prompt: str, max_tokens: int = 512) -> str:
    """로컬 LLM 호출."""
    with httpx.Client(timeout=180) as client:
        resp = client.post(
            f"{GATEWAY_URL}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # thinking 태그 제거 (Qwen3.5)
        if "<think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content


def run_phase5():
    # 1. 데이터 준비
    gt = load_ground_truth()
    chunks_path = Path(__file__).parent.parent / "data" / "prepared_chunks.json"
    with open(chunks_path) as f:
        all_chunks = json.load(f)
    logger.info(f"데이터: {len(gt)} 질문, {len(all_chunks)} 청크")

    # 2. 최적 임베딩 모델 선택
    embed_model = find_best_embedding()
    if not embed_model:
        logger.error("Phase 4 결과 없음")
        return

    # 3. 모델 로드
    logger.info("모델 로드 중...")
    ensure_models_loaded(embed_model)

    # 4. 임베딩
    logger.info(f"임베딩 ({embed_model})...")
    texts = [c["text"] for c in all_chunks]
    questions = [item["question"] for item in gt]

    with Timer("embed_chunks"):
        chunk_embeddings = get_embeddings_batch(texts, GATEWAY_URL, embed_model)
    with Timer("embed_queries"):
        query_embeddings = get_embeddings_batch(questions, GATEWAY_URL, embed_model)

    emb_array = np.array(chunk_embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_array = emb_array / norms

    logger.info(f"임베딩 완료: dim={emb_array.shape[1]}")

    # 5. LLM별 생성 + 평가
    top_k = 5
    for llm_name in LLM_MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 5: LLM={llm_name}")
        logger.info(f"{'='*60}")

        results = []
        for i, (gt_item, q_emb) in enumerate(zip(gt, query_embeddings)):
            if not q_emb or all(v == 0 for v in q_emb[:5]):
                results.append({**gt_item, "generated_answer": "", "error": "no_embedding"})
                continue

            # 검색
            q_vec = np.array(q_emb, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm == 0:
                results.append({**gt_item, "generated_answer": "", "error": "zero_norm"})
                continue
            q_vec = q_vec / norm
            scores = emb_array @ q_vec
            top_idx = np.argsort(scores)[::-1][:top_k]

            context = "\n\n---\n\n".join(all_chunks[idx]["text"] for idx in top_idx)

            # 생성
            prompt = RAG_PROMPT.format(question=gt_item["question"], context=context)
            try:
                answer = call_llm(llm_name, prompt)
            except Exception as e:
                logger.warning(f"  [{i+1}] 생성 실패: {e}")
                answer = ""

            results.append({
                "question": gt_item["question"],
                "target_answer": gt_item["target_answer"],
                "target_file": gt_item["target_file_name"],
                "domain": gt_item["domain"],
                "context_type": gt_item["context_type"],
                "generated_answer": answer,
                "retrieved_files": [all_chunks[idx]["file"] for idx in top_idx],
                "retrieved_pages": [all_chunks[idx]["page"] for idx in top_idx],
            })

            if (i + 1) % 30 == 0:
                logger.info(f"  [{i+1}/{len(gt)}] 생성 진행 중...")

        logger.info(f"  생성 완료: {sum(1 for r in results if r.get('generated_answer'))}/{len(results)}")

        # 6. LLM-as-judge 평가
        logger.info(f"  평가 중 (judge={JUDGE_MODEL})...")
        evaluated = []
        for i, r in enumerate(results):
            if not r.get("generated_answer"):
                evaluated.append({**r, "score": 0, "result": "X"})
                continue

            judge_prompt = JUDGE_PROMPT.format(
                question=r["question"],
                reference=r["target_answer"],
                generated=r["generated_answer"],
            )
            try:
                score_text = call_llm(JUDGE_MODEL, judge_prompt, max_tokens=10)
                score = int(next((c for c in score_text if c.isdigit()), "0"))
            except Exception:
                score = 0

            result = "O" if score >= 4 else "X"
            evaluated.append({**r, "score": score, "result": result})

            if (i + 1) % 30 == 0:
                o_so_far = sum(1 for e in evaluated if e["result"] == "O")
                logger.info(f"  [{i+1}/{len(gt)}] 평가 진행 중... (O: {o_so_far})")

        # 7. 집계
        o_count = sum(1 for r in evaluated if r["result"] == "O")
        total = len(evaluated)
        accuracy = o_count / total if total else 0

        by_domain = {}
        for r in evaluated:
            d = r.get("domain", "unknown")
            by_domain.setdefault(d, {"O": 0, "X": 0})
            by_domain[d][r["result"]] += 1

        logger.info(f"\n  === {llm_name} 결과 ===")
        logger.info(f"  정답률: {o_count}/{total} ({accuracy:.1%})")
        for d in sorted(by_domain):
            c = by_domain[d]
            t = c["O"] + c["X"]
            logger.info(f"    {d}: {c['O']}/{t} ({c['O']/t:.1%})")

        # 저장
        save_result("phase5_llm", llm_name, {
            "llm": llm_name,
            "embed_model": embed_model,
            "judge_model": JUDGE_MODEL,
            "top_k": top_k,
            "total": total,
            "correct": o_count,
            "accuracy": round(accuracy, 4),
            "by_domain": {d: {
                "o": c["O"],
                "total": c["O"] + c["X"],
                "accuracy": round(c["O"] / (c["O"] + c["X"]), 4),
            } for d, c in by_domain.items()},
            "results": evaluated,
        })

    # 최종 요약
    print(f"\n{'='*60}")
    print(f"  Phase 5: LLM 생성 비교")
    print(f"{'='*60}")
    print(f"  임베딩: {embed_model}")
    print(f"  Judge: {JUDGE_MODEL}")
    print(f"  ref: allganize gpt-4-turbo = 61.0% (183/300)")
    print(f"{'='*60}")
    for llm_name in LLM_MODELS:
        p = RESULTS_DIR / "phase5_llm" / f"{llm_name}.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            print(f"  {llm_name}: {d['correct']}/{d['total']} ({d['accuracy']:.1%})")


if __name__ == "__main__":
    run_phase5()
