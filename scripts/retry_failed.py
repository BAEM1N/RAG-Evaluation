#!/usr/bin/env python3
"""실패한 파일 순차 재빌드 + 재시도."""
import json, os, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from baseline_langchain import build_vectorstore, build_chain, find_pdf

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "baseline_langchain"
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4-turbo")

result_path = RESULTS_DIR / f"baseline_{LLM_MODEL}_results.json"
with open(result_path) as f:
    data = json.load(f)

failed = [r for r in data["results"] if r.get("error")]
ok = [r for r in data["results"] if not r.get("error")]
print(f"재시도 대상: {len(failed)}개 / 성공 유지: {len(ok)}개")

from collections import defaultdict
by_file = defaultdict(list)
for r in failed:
    by_file[r["target_file_name"]].append(r)

print(f"실패 파일 {len(by_file)}개")

retried = []
for fname, items in by_file.items():
    print(f"\n[{fname[:50]}] {len(items)}개 질의")
    pdf_path = find_pdf(fname)
    if not pdf_path:
        print(f"  PDF 없음")
        for item in items:
            retried.append({**item, "error": "PDF_NOT_FOUND"})
        continue
    try:
        print(f"  벡터화 시작...")
        t0 = time.time()
        vs = build_vectorstore(pdf_path)
        print(f"  벡터화 완료: {time.time()-t0:.1f}s")
        chain = build_chain(vs, LLM_MODEL)
        for item in items:
            try:
                ts = time.time()
                answer = chain.invoke(item["question"])
                new_r = {k: v for k, v in item.items() if k not in ("error", "generated_answer", "latency_sec")}
                new_r["generated_answer"] = answer
                new_r["latency_sec"] = round(time.time() - ts, 2)
                new_r["error"] = None
                retried.append(new_r)
                print(f"    OK ({time.time()-ts:.1f}s)")
            except Exception as e:
                retried.append({**item, "error": f"retry_fail: {e}"})
                print(f"    FAIL: {e}")
    except Exception as e:
        print(f"  빌드 실패: {e}")
        for item in items:
            retried.append({**item, "error": f"vs_build_fail: {e}"})

all_results = ok + retried
q_order = {r["question"]: i for i, r in enumerate(data["results"])}
all_results.sort(key=lambda r: q_order.get(r["question"], 999))

ok_count = sum(1 for r in all_results if r.get("generated_answer"))
err_count = sum(1 for r in all_results if r.get("error"))
data["results"] = all_results
data["answered"] = ok_count
data["errors"] = err_count

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n=== 최종 ===")
print(f"  응답: {ok_count}/{len(all_results)}")
print(f"  에러: {err_count}")
