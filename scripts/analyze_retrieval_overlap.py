#!/usr/bin/env python3
"""
Retrieval Cache 분석: 임베딩 간 검색 일치도 + 정답 매칭

결과:
  1. 각 임베딩의 Hit@1/5 (정답 파일/페이지 대비)
  2. 임베딩 쌍별 Jaccard overlap (검색된 (file,page) set 기준)
  3. 질문별 / 도메인별 / context_type별 분석
  4. "모든 임베딩이 같은 답" / "임베딩마다 다른 답" 분포

Usage:
  python analyze_retrieval_overlap.py               # 기본 요약
  python analyze_retrieval_overlap.py --detail      # 질문별 상세
  python analyze_retrieval_overlap.py --output csv  # CSV 출력
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "retrieval_cache"
GT_PATH = ROOT / "data" / "ground_truth.json"


def load_gt():
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    return {g["question"]: g for g in gt}


def load_all_caches():
    """모든 임베딩의 캐시 로드. dict: {embed_name: {question: retrieval}}"""
    caches = {}
    for f in sorted(CACHE_DIR.glob("*.json")):
        embed = f.stem
        with open(f, encoding="utf-8") as fh:
            items = json.load(fh)
        caches[embed] = {it["question"]: it for it in items}
    return caches


def compute_hit_metrics(caches, gt):
    """각 임베딩의 정답 매칭 지표 계산."""
    metrics = {}
    for embed, items in caches.items():
        page_hit1 = page_hit5 = file_hit1 = file_hit5 = total = 0
        for q, item in items.items():
            g = gt.get(q)
            if not g:
                continue
            target_file = g["target_file_name"]
            target_page = str(g["target_page_no"])
            files = item.get("retrieved_files", [])
            pages = [str(p) for p in item.get("retrieved_pages", [])]

            total += 1
            # file hit
            if files and files[0] == target_file:
                file_hit1 += 1
            if target_file in files[:5]:
                file_hit5 += 1
            # page hit (file + page 동시 매칭)
            pairs = list(zip(files, pages))
            if pairs and pairs[0] == (target_file, target_page):
                page_hit1 += 1
            if (target_file, target_page) in pairs[:5]:
                page_hit5 += 1
        metrics[embed] = {
            "total": total,
            "page_hit@1": round(page_hit1 / total, 4) if total else 0,
            "page_hit@5": round(page_hit5 / total, 4) if total else 0,
            "file_hit@1": round(file_hit1 / total, 4) if total else 0,
            "file_hit@5": round(file_hit5 / total, 4) if total else 0,
        }
    return metrics


def compute_pairwise_overlap(caches):
    """임베딩 쌍별 평균 Jaccard 유사도 (top-5 (file,page) set 기준)."""
    embeds = list(caches.keys())
    matrix = {}
    for a, b in combinations(embeds, 2):
        sims = []
        for q in caches[a]:
            if q not in caches[b]:
                continue
            sa = set(
                zip(
                    caches[a][q].get("retrieved_files", []),
                    [str(p) for p in caches[a][q].get("retrieved_pages", [])],
                )
            )
            sb = set(
                zip(
                    caches[b][q].get("retrieved_files", []),
                    [str(p) for p in caches[b][q].get("retrieved_pages", [])],
                )
            )
            if not sa or not sb:
                continue
            jaccard = len(sa & sb) / len(sa | sb)
            sims.append(jaccard)
        if sims:
            matrix[(a, b)] = round(sum(sims) / len(sims), 4)
    return matrix


def analyze_question_divergence(caches, gt):
    """질문별로 임베딩들의 검색 일치도 분석.
    - agreement: 상위 결과가 모든 임베딩에서 같은 비율
    - 도메인/context_type별 분포
    """
    embeds = list(caches.keys())
    any_q = list(caches[embeds[0]].keys())

    divergence = []  # (question, domain, context_type, top1_unique_count, hit_count)
    for q in any_q:
        g = gt.get(q)
        if not g:
            continue
        target = (g["target_file_name"], str(g["target_page_no"]))

        # 각 임베딩의 top-1
        top1s = []
        hits = 0
        for e in embeds:
            item = caches[e].get(q)
            if not item:
                continue
            files = item.get("retrieved_files", [])
            pages = [str(p) for p in item.get("retrieved_pages", [])]
            if files:
                top1 = (files[0], pages[0])
                top1s.append(top1)
                if top1 == target:
                    hits += 1

        divergence.append({
            "question": q[:60],
            "domain": g.get("domain", ""),
            "context_type": g.get("context_type", ""),
            "n_embed": len(top1s),
            "unique_top1": len(set(top1s)),
            "hit_count": hits,  # 정답 맞춘 임베딩 수
        })
    return divergence


def print_summary(metrics, overlap_matrix, divergence):
    # 1. Hit 메트릭 (MRR순)
    print("\n" + "=" * 75)
    print("  1. 임베딩별 Hit 메트릭 (Ground Truth 대비)")
    print("=" * 75)
    print(f"{'Embedding':<28} {'Page@1':>8} {'Page@5':>8} {'File@1':>8} {'File@5':>8}")
    print("-" * 75)
    rows = sorted(metrics.items(), key=lambda x: -x[1].get("page_hit@5", 0))
    for embed, m in rows:
        print(
            f"{embed[:28]:<28} "
            f"{m['page_hit@1']*100:>7.1f}% "
            f"{m['page_hit@5']*100:>7.1f}% "
            f"{m['file_hit@1']*100:>7.1f}% "
            f"{m['file_hit@5']*100:>7.1f}%"
        )

    # 2. Overlap 매트릭 요약
    print("\n" + "=" * 75)
    print("  2. 임베딩 쌍 평균 Jaccard Overlap (Top-5, 높을수록 유사)")
    print("=" * 75)
    sorted_overlap = sorted(overlap_matrix.items(), key=lambda x: -x[1])
    print(f"  가장 유사한 쌍 Top 10:")
    for (a, b), v in sorted_overlap[:10]:
        print(f"    {v:.4f}  {a:<25} ↔  {b}")
    print(f"\n  가장 다른 쌍 Top 10:")
    for (a, b), v in sorted_overlap[-10:]:
        print(f"    {v:.4f}  {a:<25} ↔  {b}")

    avg = sum(overlap_matrix.values()) / len(overlap_matrix) if overlap_matrix else 0
    print(f"\n  전체 평균 overlap: {avg:.4f}")

    # 3. 질문별 분산
    print("\n" + "=" * 75)
    print("  3. 질문별 Top-1 검색 일치도 분포")
    print("=" * 75)
    n_embed = divergence[0]["n_embed"] if divergence else 0
    unique_bins = defaultdict(int)
    hit_bins = defaultdict(int)
    for d in divergence:
        unique_bins[d["unique_top1"]] += 1
        hit_bins[d["hit_count"]] += 1

    total = len(divergence)
    print(f"  (임베딩 {n_embed}개 × 질문 {total}개)")
    print(f"\n  Top-1이 몇 가지로 갈리는가:")
    for k in sorted(unique_bins):
        print(f"    {k}가지: {unique_bins[k]:3d} 질문 ({unique_bins[k]/total*100:.1f}%)")

    print(f"\n  정답(target page) 맞춘 임베딩 수 분포:")
    for k in sorted(hit_bins):
        print(f"    {k}/{n_embed} 맞춤: {hit_bins[k]:3d} 질문 ({hit_bins[k]/total*100:.1f}%)")

    # 도메인별
    print(f"\n  도메인별 Top-1 다양성 (평균 unique 개수):")
    dom = defaultdict(list)
    for d in divergence:
        dom[d["domain"]].append(d["unique_top1"])
    for domain, vals in sorted(dom.items()):
        avg_u = sum(vals) / len(vals)
        print(f"    {domain:<10}: {avg_u:.2f} ({len(vals)} 질문)")

    # context_type별
    print(f"\n  Context Type별 Top-1 다양성 (평균 unique 개수):")
    ctx = defaultdict(list)
    for d in divergence:
        ctx[d["context_type"]].append(d["unique_top1"])
    for ctype, vals in sorted(ctx.items()):
        avg_u = sum(vals) / len(vals)
        print(f"    {ctype:<12}: {avg_u:.2f} ({len(vals)} 질문)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", action="store_true", help="질문별 상세 덤프")
    parser.add_argument("--output", choices=["stdout", "csv", "json"], default="stdout")
    args = parser.parse_args()

    print("[1/3] 캐시 + GT 로드 중...")
    gt = load_gt()
    caches = load_all_caches()
    print(f"  임베딩 {len(caches)}개, GT {len(gt)}개")

    print("[2/3] Hit 메트릭 계산 중...")
    metrics = compute_hit_metrics(caches, gt)

    print("[3/3] 쌍별 overlap + 질문별 분산 계산 중...")
    overlap = compute_pairwise_overlap(caches)
    divergence = analyze_question_divergence(caches, gt)

    print_summary(metrics, overlap, divergence)

    if args.detail:
        print("\n" + "=" * 75)
        print("  질문별 상세 (임베딩 일치도 낮은 순)")
        print("=" * 75)
        divergence.sort(key=lambda x: -x["unique_top1"])
        for d in divergence[:30]:
            print(
                f"  [{d['domain']:<10}/{d['context_type']:<10}] "
                f"unique={d['unique_top1']}/{d['n_embed']} hit={d['hit_count']} "
                f"| {d['question']}"
            )

    if args.output == "csv":
        out_dir = ROOT / "results" / "retrieval_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        # hit metrics
        import csv
        with open(out_dir / "hit_metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["embed", "page_hit@1", "page_hit@5", "file_hit@1", "file_hit@5", "total"])
            for e, m in sorted(metrics.items(), key=lambda x: -x[1]["page_hit@5"]):
                w.writerow([e, m["page_hit@1"], m["page_hit@5"], m["file_hit@1"], m["file_hit@5"], m["total"]])
        # overlap
        with open(out_dir / "overlap_matrix.csv", "w", newline="") as f:
            w = csv.writer(f)
            embeds = sorted(caches.keys())
            w.writerow([""] + embeds)
            for a in embeds:
                row = [a]
                for b in embeds:
                    if a == b:
                        row.append(1.0)
                    elif (a, b) in overlap:
                        row.append(overlap[(a, b)])
                    elif (b, a) in overlap:
                        row.append(overlap[(b, a)])
                    else:
                        row.append("")
                w.writerow(row)
        # divergence
        with open(out_dir / "question_divergence.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "domain", "context_type", "n_embed", "unique_top1", "hit_count"])
            for d in divergence:
                w.writerow([d["question"], d["domain"], d["context_type"], d["n_embed"], d["unique_top1"], d["hit_count"]])
        print(f"\nCSV 저장: {out_dir}/")


if __name__ == "__main__":
    main()
