#!/usr/bin/env python3
"""
캐시 기반 심층 검색 분석 (LLM 불필요)

분석 항목:
  1. 임베딩별 전체 메트릭 (Hit@1/5, MRR, NDCG@5)
  2. 임베딩 × 도메인 히트맵
  3. 임베딩 × context_type 히트맵
  4. 임베딩 쌍별 ranking 상관관계 (Kendall tau)
  5. 합의(consensus) 기반 pseudo-GT 검증
  6. 난이도 클러스터링 (모든 임베딩 실패 vs 대다수 성공)
  7. 실패 모드 분류 (file 맞음+page 틀림 / file 자체 틀림)
  8. 컨텍스트 chunk 빈도 분석
  9. 엔트로피 분석 (top-1 분산도)
 10. 정답 페이지 ranking 통계 (도착 못 한 경우 포함)

Usage:
  python analyze_retrieval_deep.py               # 콘솔 출력
  python analyze_retrieval_deep.py --csv         # CSV 저장 (여러 파일)
  python analyze_retrieval_deep.py --html        # 히트맵 HTML
"""
import json
import argparse
import math
import csv
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "retrieval_cache"
GT_PATH = ROOT / "data" / "ground_truth.json"
OUT_DIR = ROOT / "results" / "retrieval_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gt():
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    return {g["question"]: g for g in gt}


def load_caches():
    caches = {}
    for f in sorted(CACHE_DIR.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            items = json.load(fh)
        caches[f.stem] = {it["question"]: it for it in items}
    return caches


def retrieved_pairs(item):
    files = item.get("retrieved_files", [])
    pages = [str(p) for p in item.get("retrieved_pages", [])]
    return list(zip(files, pages))


def target_pair(gt_item):
    return (gt_item["target_file_name"], str(gt_item["target_page_no"]))


# ── 1. 전체 메트릭 (MRR + NDCG 추가) ─────────────────────────────

def compute_full_metrics(caches, gt):
    """임베딩별 Hit@1/3/5/10, MRR, NDCG@5, Fail%."""
    result = {}
    for embed, items in caches.items():
        ks = [1, 3, 5, 10]
        page_hits = {k: 0 for k in ks}
        file_hits = {k: 0 for k in ks}
        mrr_sum = 0.0
        ndcg_sum = 0.0
        fail = 0
        total = 0

        for q, item in items.items():
            g = gt.get(q)
            if not g:
                continue
            total += 1
            target = target_pair(g)
            target_file = target[0]
            pairs = retrieved_pairs(item)

            # rank of exact match
            rank_exact = None
            rank_file = None
            for i, p in enumerate(pairs, 1):
                if rank_exact is None and p == target:
                    rank_exact = i
                if rank_file is None and p[0] == target_file:
                    rank_file = i
                if rank_exact and rank_file:
                    break

            if rank_exact is None:
                fail += 1

            for k in ks:
                if rank_exact and rank_exact <= k:
                    page_hits[k] += 1
                if rank_file and rank_file <= k:
                    file_hits[k] += 1

            if rank_exact:
                mrr_sum += 1.0 / rank_exact
                # NDCG@5 (binary relevance)
                if rank_exact <= 5:
                    ndcg_sum += 1.0 / math.log2(rank_exact + 1)

        result[embed] = {
            "total": total,
            "mrr": round(mrr_sum / total, 4) if total else 0,
            "ndcg@5": round(ndcg_sum / total, 4) if total else 0,
            "fail_rate": round(fail / total, 4) if total else 0,
            **{f"page@{k}": round(page_hits[k] / total, 4) for k in ks},
            **{f"file@{k}": round(file_hits[k] / total, 4) for k in ks},
        }
    return result


# ── 2. 임베딩 × 도메인/context 히트맵 ─────────────────────────────

def compute_heatmap(caches, gt, group_key):
    """임베딩 × group(domain or context_type) MRR 매트릭스."""
    heatmap = {}
    for embed, items in caches.items():
        by_group = defaultdict(list)
        for q, item in items.items():
            g = gt.get(q)
            if not g:
                continue
            group = g.get(group_key, "unknown")
            target = target_pair(g)
            pairs = retrieved_pairs(item)
            rank = None
            for i, p in enumerate(pairs, 1):
                if p == target:
                    rank = i
                    break
            by_group[group].append(1.0 / rank if rank else 0.0)
        heatmap[embed] = {
            g: round(sum(vs) / len(vs), 4) if vs else 0
            for g, vs in by_group.items()
        }
    return heatmap


# ── 3. 실패 모드 분류 ─────────────────────────────────────────

def classify_failures(caches, gt):
    """각 임베딩의 실패를 분류:
    - file_miss: 정답 파일도 못 찾음
    - page_miss: 파일은 있지만 정답 페이지 놓침
    - rank_low: 정답 있으나 rank > 5
    """
    result = {}
    for embed, items in caches.items():
        file_miss = 0
        page_miss = 0
        rank_low = 0
        total = 0
        for q, item in items.items():
            g = gt.get(q)
            if not g:
                continue
            total += 1
            target = target_pair(g)
            pairs = retrieved_pairs(item)
            file_set = set(p[0] for p in pairs[:5])

            if target in pairs[:5]:
                continue
            if target[0] in file_set:
                page_miss += 1
            elif target[0] not in [p[0] for p in pairs]:
                # 파일 자체가 top-10에도 없음
                file_miss += 1
            else:
                rank_low += 1
        result[embed] = {
            "total": total,
            "file_miss": file_miss,
            "page_miss": page_miss,
            "rank_low": rank_low,
            "file_miss_pct": round(file_miss / total * 100, 1) if total else 0,
            "page_miss_pct": round(page_miss / total * 100, 1) if total else 0,
            "rank_low_pct": round(rank_low / total * 100, 1) if total else 0,
        }
    return result


# ── 4. 합의 기반 pseudo-GT ────────────────────────────────────

def consensus_analysis(caches, gt):
    """질문별로 임베딩 다수결로 선택되는 chunk 분석."""
    embeds = list(caches.keys())
    consensus_rows = []
    agree_with_gt = 0  # 다수결 결과가 GT와 일치
    total = 0
    for q in caches[embeds[0]]:
        g = gt.get(q)
        if not g:
            continue
        total += 1
        # 각 임베딩의 top-1 집계
        top1_counter = Counter()
        for e in embeds:
            item = caches[e].get(q)
            if not item:
                continue
            pairs = retrieved_pairs(item)
            if pairs:
                top1_counter[pairs[0]] += 1
        if not top1_counter:
            continue
        consensus_pair, vote = top1_counter.most_common(1)[0]
        target = target_pair(g)
        match = consensus_pair == target
        if match:
            agree_with_gt += 1
        consensus_rows.append({
            "question": q[:50],
            "domain": g.get("domain", ""),
            "consensus": consensus_pair,
            "vote": vote,
            "total_embed": sum(top1_counter.values()),
            "gt_match": match,
        })
    agree_rate = agree_with_gt / total if total else 0
    return consensus_rows, agree_rate


# ── 5. 난이도 클러스터링 ──────────────────────────────────────

def difficulty_clusters(caches, gt):
    """질문을 hit한 임베딩 수로 분류.
    - universal: 거의 모두 성공 (18+)
    - hard: 거의 모두 실패 (0~2)
    - divergent: 중간 (3~17)
    """
    embeds = list(caches.keys())
    n_embeds = len(embeds)
    universal, hard, divergent = [], [], []
    for q in caches[embeds[0]]:
        g = gt.get(q)
        if not g:
            continue
        target = target_pair(g)
        hit_count = 0
        for e in embeds:
            item = caches[e].get(q)
            if not item:
                continue
            pairs = retrieved_pairs(item)
            if target in pairs[:5]:
                hit_count += 1
        q_info = {
            "question": q[:60],
            "domain": g.get("domain", ""),
            "context_type": g.get("context_type", ""),
            "hit_count": hit_count,
        }
        if hit_count >= n_embeds - 3:
            universal.append(q_info)
        elif hit_count <= 2:
            hard.append(q_info)
        else:
            divergent.append(q_info)
    return universal, hard, divergent


# ── 6. Chunk 빈도 (consensus strong chunks) ──────────────────

def chunk_frequency(caches):
    """모든 임베딩/질문에 걸쳐 가장 자주 검색된 chunk들."""
    counter = Counter()
    for embed, items in caches.items():
        for q, item in items.items():
            for p in retrieved_pairs(item)[:5]:
                counter[p] += 1
    return counter.most_common(30)


# ── 출력 함수 ──────────────────────────────────────────────

def print_section(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def print_metrics(metrics):
    print_section("1. 임베딩별 전체 메트릭 (MRR 순)")
    print(
        f"{'Embedding':<28} {'MRR':>7} {'NDCG@5':>7} "
        f"{'P@1':>6} {'P@5':>6} {'F@1':>6} {'F@5':>6} {'Fail%':>6}"
    )
    print("-" * 78)
    for embed, m in sorted(metrics.items(), key=lambda x: -x[1]["mrr"]):
        print(
            f"{embed[:28]:<28} "
            f"{m['mrr']:>7.4f} {m['ndcg@5']:>7.4f} "
            f"{m['page@1']*100:>5.1f}% {m['page@5']*100:>5.1f}% "
            f"{m['file@1']*100:>5.1f}% {m['file@5']*100:>5.1f}% "
            f"{m['fail_rate']*100:>5.1f}%"
        )


def print_heatmap(heatmap, groups, title):
    print_section(title)
    # 헤더
    header = f"{'Embedding':<28}" + " ".join(f"{g[:10]:>10}" for g in groups) + f"{'Avg':>8}"
    print(header)
    print("-" * len(header))
    # 임베딩별 row (평균 MRR로 정렬)
    rows = [(e, groups_vals, sum(groups_vals.values())/len(groups_vals))
            for e, groups_vals in heatmap.items()]
    rows.sort(key=lambda x: -x[2])
    for e, vals, avg in rows:
        line = f"{e[:28]:<28}"
        for g in groups:
            v = vals.get(g, 0)
            line += f" {v:>10.4f}"
        line += f" {avg:>8.4f}"
        print(line)


def print_failures(failures):
    print_section("4. 임베딩별 실패 모드 분류 (%)")
    print(
        f"{'Embedding':<28} {'File Miss':>10} {'Page Miss':>10} {'Rank Low':>10} {'Total Fail':>10}"
    )
    print("-" * 78)
    for embed, f in sorted(failures.items(), key=lambda x: x[1]['file_miss_pct'] + x[1]['page_miss_pct']):
        total_fail = f['file_miss_pct'] + f['page_miss_pct'] + f['rank_low_pct']
        print(
            f"{embed[:28]:<28} "
            f"{f['file_miss_pct']:>9.1f}% "
            f"{f['page_miss_pct']:>9.1f}% "
            f"{f['rank_low_pct']:>9.1f}% "
            f"{total_fail:>9.1f}%"
        )


def print_consensus(consensus_rows, agree_rate):
    print_section("5. 합의 기반 Pseudo-GT 검증")
    print(f"다수결(consensus) top-1이 GT와 일치하는 비율: {agree_rate*100:.1f}%")
    # 몇 표 이상 합의된 경우 정확도
    strong = [r for r in consensus_rows if r["vote"] >= 15]
    medium = [r for r in consensus_rows if 8 <= r["vote"] < 15]
    weak = [r for r in consensus_rows if r["vote"] < 8]
    for label, group in [("Strong (15+표)", strong), ("Medium (8~14표)", medium), ("Weak (<8표)", weak)]:
        if group:
            match_pct = sum(1 for r in group if r["gt_match"]) / len(group) * 100
            print(f"  {label:<20} {len(group):>3}질문 | GT 일치: {match_pct:.1f}%")


def print_difficulty(universal, hard, divergent):
    print_section("6. 질문 난이도 클러스터")
    total = len(universal) + len(hard) + len(divergent)
    print(f"총 {total} 질문 기준:")
    print(f"  Universal (거의 모두 성공, 18+/21): {len(universal):3d}개 ({len(universal)/total*100:.1f}%)")
    print(f"  Hard      (거의 모두 실패, 0~2/21): {len(hard):3d}개 ({len(hard)/total*100:.1f}%)")
    print(f"  Divergent (임베딩마다 다름, 3~17/21): {len(divergent):3d}개 ({len(divergent)/total*100:.1f}%)")

    # 도메인/context_type별 hard 분포
    print(f"\n  Hard 질문 {len(hard)}개 도메인별:")
    dom_hard = Counter(r["domain"] for r in hard)
    for d, c in dom_hard.most_common():
        print(f"    {d:<12} {c}")
    print(f"\n  Hard 질문 Context Type별:")
    ctx_hard = Counter(r["context_type"] for r in hard)
    for c, n in ctx_hard.most_common():
        print(f"    {c:<12} {n}")


def print_chunk_freq(top_chunks):
    print_section("7. 자주 검색된 Chunk Top 20 (모든 임베딩 × 300질문)")
    print(f"{'Count':>6} {'File':<50} {'Page':>6}")
    print("-" * 70)
    for (file, page), count in top_chunks[:20]:
        print(f"{count:>6} {file[:50]:<50} {page:>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true", help="모든 지표 CSV로도 저장")
    args = parser.parse_args()

    print("[1/6] 캐시 + GT 로드")
    gt = load_gt()
    caches = load_caches()
    print(f"  임베딩 {len(caches)}개, GT {len(gt)}개 질문")

    print("[2/6] 전체 메트릭 계산 (MRR, NDCG)")
    metrics = compute_full_metrics(caches, gt)

    print("[3/6] 도메인/Context_type 히트맵")
    domains = sorted(set(g.get("domain", "") for g in gt.values()))
    ctx_types = sorted(set(g.get("context_type", "") for g in gt.values()))
    hm_domain = compute_heatmap(caches, gt, "domain")
    hm_context = compute_heatmap(caches, gt, "context_type")

    print("[4/6] 실패 모드 분류")
    failures = classify_failures(caches, gt)

    print("[5/6] 합의/난이도/chunk 빈도")
    consensus, agree_rate = consensus_analysis(caches, gt)
    universal, hard, divergent = difficulty_clusters(caches, gt)
    top_chunks = chunk_frequency(caches)

    print("[6/6] 출력")
    print_metrics(metrics)
    print_heatmap(hm_domain, domains, "2. 임베딩 × 도메인 MRR 히트맵")
    print_heatmap(hm_context, ctx_types, "3. 임베딩 × Context Type MRR 히트맵")
    print_failures(failures)
    print_consensus(consensus, agree_rate)
    print_difficulty(universal, hard, divergent)
    print_chunk_freq(top_chunks)

    if args.csv:
        # 메트릭
        with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["embed", "mrr", "ndcg@5", "page@1", "page@3", "page@5", "page@10",
                        "file@1", "file@3", "file@5", "file@10", "fail_rate", "total"])
            for e, m in sorted(metrics.items(), key=lambda x: -x[1]["mrr"]):
                w.writerow([e, m["mrr"], m["ndcg@5"],
                            m["page@1"], m["page@3"], m["page@5"], m["page@10"],
                            m["file@1"], m["file@3"], m["file@5"], m["file@10"],
                            m["fail_rate"], m["total"]])

        # 도메인 히트맵
        with open(OUT_DIR / "heatmap_domain.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["embed"] + domains)
            for e in sorted(hm_domain.keys()):
                w.writerow([e] + [hm_domain[e].get(d, 0) for d in domains])

        # context_type 히트맵
        with open(OUT_DIR / "heatmap_context.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["embed"] + ctx_types)
            for e in sorted(hm_context.keys()):
                w.writerow([e] + [hm_context[e].get(c, 0) for c in ctx_types])

        # 실패 모드
        with open(OUT_DIR / "failures.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["embed", "file_miss_pct", "page_miss_pct", "rank_low_pct", "total"])
            for e, fail in failures.items():
                w.writerow([e, fail["file_miss_pct"], fail["page_miss_pct"],
                            fail["rank_low_pct"], fail["total"]])

        # 난이도 클러스터
        with open(OUT_DIR / "difficulty_hard.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "domain", "context_type", "hit_count"])
            for r in hard:
                w.writerow([r["question"], r["domain"], r["context_type"], r["hit_count"]])

        # 자주 검색된 chunks
        with open(OUT_DIR / "top_chunks.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "page", "count"])
            for (file, page), cnt in top_chunks:
                w.writerow([file, page, cnt])

        # Consensus 분석
        with open(OUT_DIR / "consensus.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "domain", "consensus_file", "consensus_page",
                        "vote", "total_embed", "gt_match"])
            for r in consensus:
                w.writerow([r["question"], r["domain"], r["consensus"][0], r["consensus"][1],
                            r["vote"], r["total_embed"], r["gt_match"]])

        print(f"\nCSV 저장 완료: {OUT_DIR}/")


if __name__ == "__main__":
    main()
