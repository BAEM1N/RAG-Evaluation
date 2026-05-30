#!/usr/bin/env python3
"""
gpt-5.4-pro 생성 batch 완료를 폴링하다가, 완료되면 자동으로:
  1) gpt-5.4-pro 결과 다운로드 (Responses API output 파싱)
  2) 모든 candidate generation을 통일 형식으로 정리
  3) Flagship judge batch 제출
     - Anthropic claude-opus-4-7 (cache 90% off)
     - Google gemini-3.1-pro-preview (context cache)
     - (OpenAI gpt-5.4-pro judge는 Responses API 모듈 패치 후 별도 제출)

실행:
  nohup python scripts/auto_judge_pipeline.py > logs/auto_judge.log 2>&1 &
"""
import os, json, time, sys, subprocess, re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"), override=True)
from openai import OpenAI

QID_RE = re.compile(r'q\d{3}')
CUSTOM_ID_SAFE = re.compile(r'[^a-zA-Z0-9_-]')

def _normalize_qid(raw, idx):
    if raw:
        m = QID_RE.search(str(raw))
        if m:
            return m.group()
    return f"q{idx:03d}"

def _safe_custom_id(model_id, qid, metric):
    raw = f"{model_id}__{qid}__{metric}"
    safe = CUSTOM_ID_SAFE.sub('_', raw)
    return safe[:64]

ROOT = Path(".")
PROVIDER_DIR = ROOT / "results/phase5_exp_b_provider"
LOCAL_DIR = ROOT / "results/phase5_exp_b_llm"
JUDGE_DIR = ROOT / "results/phase5_judge_flagship"
JUDGE_DIR.mkdir(exist_ok=True)
LOG = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ============== STEP 1: gpt-5.4-pro 완료 대기 ==============
def wait_gpt54_pro():
    LOG("Step 1: gpt-5.4-pro 완료 대기")
    oai = OpenAI()
    while True:
        for b in oai.batches.list(limit=20).data:
            try:
                fi = oai.files.retrieve(b.input_file_id)
                fn = fi.filename or ""
            except Exception:
                continue
            if "gpt-5.4-pro_responses" not in fn:
                continue
            r = b.request_counts
            LOG(f"  gpt-5.4-pro batch {b.status} {r.completed}/{r.total}")
            if b.status in ("completed", "failed", "expired", "cancelled"):
                return b
            break
        time.sleep(120)


# ============== STEP 2: gpt-5.4-pro Responses 다운로드 ==============
def download_gpt54_pro(b):
    LOG("Step 2: gpt-5.4-pro 결과 다운로드 + 파싱")
    if not b.output_file_id:
        LOG("  ⚠️ output_file_id 없음 — skip")
        return
    oai = OpenAI()
    raw = oai.files.content(b.output_file_id).text
    rows = []
    for line in raw.strip().split("\n"):
        d = json.loads(line)
        cid = d.get("custom_id", "")
        body = d.get("response", {}).get("body", {})
        # Responses API 출력 구조: output[].content[].text
        text = ""
        for o in body.get("output", []):
            if o.get("type") == "message":
                for c in o.get("content", []):
                    if c.get("type") == "output_text":
                        text += c.get("text", "")
        usage = body.get("usage", {})
        rows.append({
            "qid": cid,
            "generated_answer": text,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        })
    out = {
        "provider": "openai", "llm": "gpt-5.4-pro",
        "embed_model": "gemma-embed-300m", "top_k": 5,
        "total": len(rows), "answered": sum(1 for r in rows if r["generated_answer"]),
        "results": rows,
    }
    out_path = PROVIDER_DIR / "expB__gemma-embed-300m__gpt-5.4-pro.json"
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    LOG(f"  ✅ {out_path.name}: {out['answered']}/{out['total']}")


# ============== STEP 3: 모든 candidate 통일 형식 ==============
def collect_all_candidates():
    """45-46개 LLM 결과를 (qid → {model, answer}) 형태로 수집."""
    LOG("Step 3: 모든 candidate 수집")
    candidates = []  # [{model_id, source, qid, generated_answer}]

    def add_from_json(path, source):
        try:
            d = json.load(open(path))
        except Exception as e:
            LOG(f"  ❌ {path.name}: {e}")
            return 0
        rows = d.get("results", [])
        # 파일 stem 기준 (think/nothink 변형 분리 보장)
        model = path.stem.replace("expB__gemma-embed-300m__", "")
        n = 0
        for idx, r in enumerate(rows):
            qid = _normalize_qid(r.get("qid") or r.get("custom_id"), idx)
            ans = r.get("generated_answer") or r.get("answer") or r.get("content")
            if not ans:
                continue
            candidates.append({"model_id": model, "source": source, "qid": qid, "generated_answer": ans})
            n += 1
        return n

    def add_from_jsonl(path, source):
        n = 0
        seen = set()
        for idx, line in enumerate(open(path)):
            try:
                r = json.loads(line)
                qid = _normalize_qid(r.get("qid"), idx)
                ans = r.get("answer") or r.get("generated_answer")
                model = path.stem.replace("expB__gemma-embed-300m__", "").replace("_", "/", 1)
                if not (ans and qid not in seen):
                    continue
                seen.add(qid)
                candidates.append({"model_id": model, "source": source, "qid": qid, "generated_answer": ans})
                n += 1
            except: pass
        return n

    # Local 12
    for f in sorted(LOCAL_DIR.glob("expB__*.json")):
        n = add_from_json(f, "local")
        LOG(f"  local: {f.name} ({n})")
    # Provider .json (Anthropic + OpenAI + Google)
    for f in sorted(PROVIDER_DIR.glob("expB__*.json")):
        if "dryrun" in f.name or "damaged" in f.name or "before_rebuild" in f.name:
            continue
        n = add_from_json(f, "provider-batch")
        LOG(f"  provider-batch: {f.name} ({n})")
    # Provider .jsonl (OpenRouter realtime)
    for f in sorted(PROVIDER_DIR.glob("expB__*.jsonl")):
        n = add_from_jsonl(f, "openrouter")
        LOG(f"  openrouter: {f.name} ({n})")

    # 모델별 집계
    from collections import Counter
    model_counts = Counter(c["model_id"] for c in candidates)
    LOG(f"  총 candidates: {len(candidates)} ({len(model_counts)} models)")
    return candidates


# ============== STEP 4: judge 입력 빌드 + 제출 ==============
def submit_judges(candidates):
    LOG("Step 4: 3-Judge batch 제출")
    # ground truth
    gt = json.load(open(ROOT / "data/ground_truth_filtered.json"))
    qid_to_target = {f"q{i:03d}": item for i, item in enumerate(gt)}

    # 입력 매트릭스 빌드
    items = []  # {custom_id, qid, metric, candidate_model, question, target, candidate}
    metrics = ["similarity", "correctness", "completeness", "faithfulness"]
    for c in candidates:
        qid = c["qid"]
        gt_item = qid_to_target.get(qid)
        if not gt_item:
            continue
        for m in metrics:
            items.append({
                "custom_id": _safe_custom_id(c['model_id'], qid, m),
                "qid": qid, "metric": m,
                "candidate_model": c["model_id"],
                "question": gt_item.get("question") or gt_item.get("query"),
                "target": gt_item.get("target_answer") or gt_item.get("answer"),
                "candidate": c["generated_answer"],
            })
    LOG(f"  judge input items: {len(items)}")

    # Anthropic
    sys.path.insert(0, str(ROOT / "scripts"))
    from providers import anthropic_batch
    LOG("  Anthropic claude-opus-4-7 batch 제출 ...")
    requests = anthropic_batch.write_judge_batch_requests(items, judge_model="claude-opus-4-7")
    batch_ids_anth = anthropic_batch.submit_batch(
        requests,
        save_path=JUDGE_DIR / "_inputs/anthropic_judge.jsonl",
        chunk_size=5000,
    )
    LOG(f"    Anthropic batch_ids ({len(batch_ids_anth)} chunks): {batch_ids_anth}")

    # Gemini
    from providers import gemini_batch
    LOG("  Gemini gemini-3.1-pro-preview cache + batch 제출 ...")
    cache_map = {}
    unique_qm = list({(it["qid"], it["metric"]): it for it in items}.values())
    LOG(f"    Gemini caches 생성 ({len(unique_qm)})...")
    for it in unique_qm:
        try:
            name = gemini_batch.create_judge_cache(
                question=it["question"], target=it["target"],
                metric=it["metric"], model="gemini-3.1-pro-preview",
            )
            cache_map[(it["qid"], it["metric"])] = name
        except Exception as e:
            LOG(f"      cache fail (qid={it['qid']}, m={it['metric']}): {str(e)[:80]}")
    LOG(f"    cache 생성: {len(cache_map)}/{len(unique_qm)}")
    jsonl = JUDGE_DIR / "_inputs/gemini_judge.jsonl"
    jsonl.parent.mkdir(exist_ok=True, parents=True)
    gemini_batch.write_judge_batch_jsonl_with_cache(items, cache_map, judge_model="gemini-3.1-pro-preview", out_path=jsonl)
    batch_id_g = gemini_batch.submit_batch(jsonl, model="gemini-3.1-pro-preview", display_name="phase5-judge-gemini-3.1-pro", task_type="judge")
    LOG(f"    Gemini batch_id: {batch_id_g}")

    # OpenAI gpt-5.4-pro judge는 Responses API patch가 필요해서 일단 skip — 별도 처리
    LOG("  OpenAI gpt-5.4-pro judge: Responses API patch 후 별도 제출 (skip)")

    # registry
    reg_path = JUDGE_DIR / "_batch_registry.json"
    json.dump({
        "anthropic": {"claude-opus-4-7": {"batch_ids": batch_ids_anth, "status": "submitted",
                                           "submitted_at": datetime.now().isoformat()}},
        "google":    {"gemini-3.1-pro-preview": {"batch_id": batch_id_g, "status": "submitted",
                                                  "submitted_at": datetime.now().isoformat()}},
    }, open(reg_path, "w"), indent=2)
    LOG(f"  registry: {reg_path}")


# ============== MAIN ==============
def main():
    skip_wait = os.environ.get("SKIP_WAIT") == "1"
    if not skip_wait:
        b = wait_gpt54_pro()
        if b.status == "completed":
            download_gpt54_pro(b)
    else:
        LOG("SKIP_WAIT=1 → generation 완료 가정, candidate 수집부터 시작")
    candidates = collect_all_candidates()
    submit_judges(candidates)
    LOG("=== 자동 trigger 완료 ===")


if __name__ == "__main__":
    main()
