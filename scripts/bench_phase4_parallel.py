#!/usr/bin/env python3
"""
Phase 4: Embedding 모델 비교 (병렬 버전)

VRAM 여유분에 맞춰 여러 모델을 동시 로드 → 병렬 임베딩 → 일괄 언로드.
소형 모델(~1GB)은 10개씩, 중형은 3~4개씩, 대형은 1개씩 처리.
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
logger = logging.getLogger("phase4-parallel")

GATEWAY_URL = "http://192.168.50.245:8080/v1"
GATEWAY_BASE = "http://192.168.50.245:8080"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"

EMBEDDING_MODELS = {
    # 대형 (단독 실행)
    "qwen3-embed-8b":           {"vram": 9.0, "dim": 4096},
    "llama-embed-nemotron-8b":  {"vram": 9.0, "dim": 4096},
    "nemotron-embed-8b":        {"vram": 9.0, "dim": 4096},
    "e5-mistral-7b":            {"vram": 8.0, "dim": 4096},
    # 중형
    "qwen3-embed-4b":           {"vram": 5.0, "dim": 4096},
    "jina-v4-retrieval":        {"vram": 4.0, "dim": 4096},
    "jina-v4-code":             {"vram": 4.0, "dim": 4096},
    "jina-code-1.5b":           {"vram": 2.0, "dim": 1024},
    # 한국어 특화 소형
    "snowflake-arctic-ko":      {"vram": 1.0, "dim": 1024},
    "pixie-rune-v1":            {"vram": 1.0, "dim": 1024},
    "kure-v1":                  {"vram": 1.0, "dim": 1024},
    "koe5":                     {"vram": 1.0, "dim": 1024},
    # 다국어 소형
    "qwen3-embed-0.6b":         {"vram": 1.0, "dim": 1024},
    "snowflake-arctic-l-v2":    {"vram": 1.0, "dim": 1024},
    "bge-m3":                   {"vram": 1.0, "dim": 1024},
    "me5-large-instruct":       {"vram": 1.0, "dim": 1024},
    "jina-v5-small-retrieval":  {"vram": 1.0, "dim": 1024},
    "harrier-0.6b":             {"vram": 0.8, "dim": 1024},
    "labse":                    {"vram": 1.0, "dim": 768},
    "nomic-embed-v2-moe":       {"vram": 1.0, "dim": 768},
    # 초소형
    "mxbai-embed-large":        {"vram": 0.5, "dim": 1024},
    "voyage-4-nano":            {"vram": 0.5, "dim": 1024},
    "gemma-embed-300m":         {"vram": 0.5, "dim": 768},
    "granite-278m":             {"vram": 0.5, "dim": 768},
    "harrier-270m":             {"vram": 0.3, "dim": 1024},
    "jina-v5-nano-matching":    {"vram": 0.5, "dim": 512},
    "granite-107m":             {"vram": 0.2, "dim": 768},
    # 초대형 (단독)
    "harrier-27b":              {"vram": 20.0, "dim": 4096},
}

# VRAM 예산: 상시 모델(60GB) 제외한 여유분
# qwen3-embed-8b(9GB)는 언로드 가능 → 최대 36+9=45GB 사용 가능
VRAM_BUDGET = 36.0


def find_pdf(filename):
    for domain_dir in PDF_DIR.iterdir():
        if domain_dir.is_dir():
            pdf = domain_dir / filename
            if pdf.exists():
                return pdf
    return None


def group_models_by_vram(models: dict, budget: float) -> list:
    """VRAM 예산에 맞게 모델을 배치 그룹으로 나누기."""
    # 이미 완료된 모델 제외
    done_dir = RESULTS_DIR / "phase4_embedding"
    done_dir.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in done_dir.glob("*.json")}

    remaining = {k: v for k, v in models.items() if k not in done}
    if done:
        logger.info(f"이미 완료: {len(done)}개 ({', '.join(sorted(done))})")
    logger.info(f"남은 모델: {len(remaining)}개")

    # VRAM 크기순 정렬 (작은 것부터 → 많이 묶을 수 있음)
    sorted_models = sorted(remaining.items(), key=lambda x: x[1]["vram"])

    groups = []
    current_group = []
    current_vram = 0.0

    for name, info in sorted_models:
        if current_vram + info["vram"] <= budget and len(current_group) < 12:
            current_group.append(name)
            current_vram += info["vram"]
        else:
            if current_group:
                groups.append(current_group)
            current_group = [name]
            current_vram = info["vram"]

    if current_group:
        groups.append(current_group)

    return groups


def load_models(model_names: list) -> dict:
    """게이트웨이에 여러 모델 동시 로드."""
    with httpx.Client(timeout=120) as client:
        # 현재 실행 중인 임베딩 모델 확인
        resp = client.get(f"{GATEWAY_BASE}/v1/models")
        data = resp.json()
        running_embeds = {m["alias"] for m in data["running"] if m["mode"] == "embedding"}

        # 이미 로드된 모델 중 이번 그룹에 없는 것 언로드
        for alias in running_embeds:
            if alias not in model_names:
                try:
                    client.post(f"{GATEWAY_BASE}/v1/models/unload", json={"model": alias})
                    logger.info(f"  언로드: {alias}")
                except Exception:
                    pass

        # 새 모델 로드
        loaded = {}
        for name in model_names:
            if name in running_embeds:
                loaded[name] = True
                continue
            try:
                resp = client.post(f"{GATEWAY_BASE}/v1/models/load", json={"model": name})
                if resp.status_code == 200:
                    loaded[name] = True
                    logger.info(f"  로드: {name}")
                else:
                    logger.error(f"  로드 실패: {name} → {resp.status_code}")
            except Exception as e:
                logger.error(f"  로드 실패: {name} → {e}")

        # 헬스체크 대기
        time.sleep(3)
        for _ in range(30):
            try:
                r = client.get(f"{GATEWAY_BASE}/v1/models")
                alive = {m["alias"] for m in r.json()["running"] if m["alive"] and m["mode"] == "embedding"}
                if all(n in alive for n in loaded):
                    return loaded
            except Exception:
                pass
            time.sleep(2)

    return loaded


def embed_with_model(model_name: str, texts: list, questions: list) -> dict:
    """단일 모델로 청크 + 쿼리 임베딩."""
    t0 = time.perf_counter()
    chunk_embeddings = get_embeddings_batch(texts, GATEWAY_URL, model_name)
    embed_time = time.perf_counter() - t0

    query_embeddings = get_embeddings_batch(questions, GATEWAY_URL, model_name)

    return {
        "chunk_embeddings": chunk_embeddings,
        "query_embeddings": query_embeddings,
        "embed_time": embed_time,
    }


def evaluate_model(
    model_name: str,
    model_info: dict,
    gt: list,
    chunks: list,
    emb_result: dict,
) -> dict:
    """모델 평가 결과 계산."""
    chunk_embeddings = emb_result["chunk_embeddings"]
    query_embeddings = emb_result["query_embeddings"]

    emb_array = np.array(chunk_embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_array = emb_array / norms

    results = []
    for gt_item, q_emb in zip(gt, query_embeddings):
        if not q_emb or all(v == 0 for v in q_emb[:5]):
            continue

        q_vec = np.array(q_emb, dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm == 0:
            continue
        q_vec = q_vec / norm

        scores = emb_array @ q_vec
        top_idx = np.argsort(scores)[::-1][:10]

        retrieved = []
        for rank, idx in enumerate(top_idx, 1):
            retrieved.append({
                "file": chunks[idx]["file"],
                "page": chunks[idx]["page"],
                "rank": rank,
                "score": float(scores[idx]),
            })

        results.append({
            "question": gt_item["question"],
            "target_file": gt_item["target_file_name"],
            "target_page": gt_item["target_page_no"],
            "domain": gt_item["domain"],
            "context_type": gt_item["context_type"],
            "retrieved": retrieved,
        })

    metrics = compute_retrieval_metrics(results)
    embed_speed = len(chunks) / emb_result["embed_time"] if emb_result["embed_time"] > 0 else 0
    metrics["embed_speed_chunks_per_sec"] = round(embed_speed, 1)
    metrics["dim"] = model_info["dim"]
    metrics["vram_gb"] = model_info["vram"]

    return {
        "model": model_name,
        "dim": model_info["dim"],
        "vram_gb": model_info["vram"],
        "total_chunks": len(chunks),
        "embed_time_sec": round(emb_result["embed_time"], 2),
        "embed_speed": round(embed_speed, 1),
        "metrics": metrics,
        "by_domain": compute_metrics_by_group(results, "domain"),
        "by_context_type": compute_metrics_by_group(results, "context_type"),
    }


def process_model_group(
    group: list,
    gt: list,
    chunks: list,
    texts: list,
    questions: list,
):
    """모델 그룹 병렬 처리."""
    group_vram = sum(EMBEDDING_MODELS[m]["vram"] for m in group)
    logger.info(f"\n{'='*60}")
    logger.info(f"그룹 처리: {group} (VRAM: {group_vram:.1f}GB)")
    logger.info(f"{'='*60}")

    # 모델 로드
    loaded = load_models(group)
    if not loaded:
        logger.error("모델 로드 실패")
        return {}

    # 병렬 임베딩
    results = {}
    with ThreadPoolExecutor(max_workers=len(group)) as executor:
        futures = {
            executor.submit(embed_with_model, name, texts, questions): name
            for name in group if name in loaded
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                emb_result = future.result()
                # 평가
                model_result = evaluate_model(
                    name, EMBEDDING_MODELS[name], gt, chunks, emb_result
                )
                save_result("phase4_embedding", name, model_result)
                m = model_result["metrics"]
                logger.info(
                    f"  {name}: MRR={m['mrr']:.4f} Hit@5={m.get('page_hit@5',0):.1%} "
                    f"({model_result['embed_speed']:.0f} chunks/sec)"
                )
                results[name] = m
            except Exception as e:
                logger.error(f"  {name} 실패: {e}")
                results[name] = {"error": str(e)}

    return results


def main():
    gt = load_ground_truth()

    # 청크 준비: 사전 생성된 JSON이 있으면 사용 (원격 실행 대응)
    prepared_path = Path(__file__).parent.parent / "data" / "prepared_chunks.json"
    if prepared_path.exists():
        logger.info(f"사전 준비된 청크 로딩: {prepared_path}")
        with open(prepared_path, encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        logger.info("청크 준비 중 (PDF 파싱)...")
        all_chunks = []
        parsed_cache = {}
        for gt_item in gt:
            fname = gt_item["target_file_name"]
            if fname not in parsed_cache:
                pdf_path = find_pdf(fname)
                if pdf_path:
                    parsed_cache[fname] = parse_pdf_pymupdf4llm(str(pdf_path))

        for fname, pages in parsed_cache.items():
            chunks = chunk_pages(pages, 500, 100)
            for c in chunks:
                c["file"] = fname
            all_chunks.extend(chunks)

    texts = [c["text"] for c in all_chunks]
    questions = [item["question"] for item in gt]
    logger.info(f"청크: {len(all_chunks)}개, 질문: {len(questions)}개")

    # 모델 그룹핑
    groups = group_models_by_vram(EMBEDDING_MODELS, VRAM_BUDGET)
    logger.info(f"\n모델 그룹 ({len(groups)}개 배치):")
    for i, g in enumerate(groups):
        vram = sum(EMBEDDING_MODELS[m]["vram"] for m in g)
        logger.info(f"  배치 {i+1}: {g} ({vram:.1f}GB)")

    # 그룹별 처리
    all_metrics = {}
    for i, group in enumerate(groups):
        logger.info(f"\n[배치 {i+1}/{len(groups)}]")
        metrics = process_model_group(group, gt, all_chunks, texts, questions)
        all_metrics.update(metrics)

    # 최종 요약
    print(f"\n{'='*70}")
    print(f"  Phase 4: Embedding 비교 (병렬)")
    print(f"{'='*70}")
    print("%-28s %8s %8s %8s %6s %8s" % ("Model", "Hit@1", "Hit@5", "MRR", "dim", "spd"))
    print("-" * 70)
    for name, m in sorted(all_metrics.items(), key=lambda x: x[1].get("mrr", 0), reverse=True):
        if "error" in m:
            print(f"%-28s {'ERROR':>8s}" % name)
        else:
            print("%-28s %7.1f%% %7.1f%% %8.4f %6d %7.0f" % (
                name[:28],
                m.get("page_hit@1", 0) * 100,
                m.get("page_hit@5", 0) * 100,
                m.get("mrr", 0),
                m.get("dim", 0),
                m.get("embed_speed_chunks_per_sec", 0),
            ))


if __name__ == "__main__":
    main()
