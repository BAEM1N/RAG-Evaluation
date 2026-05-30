#!/usr/bin/env python3
"""Open Weights vs Closed Weights 답변 생성 모델 비교 분석 + 시각화.

목표: GPT-5.4 / GPT-5.4-pro 같은 Closed 모델을 대체할 수 있는 Open Weights 로컬 모델 발굴.

산출물 (docs/images/):
  fig1_open_vs_closed_scatter.png  답변 생성 모델별 (Closed-pool acc, Open-pool acc) 산점도
  fig2_alternative_to_gpt54.png    GPT-5.4-pro 기준 대체 가능 후보 ranking
  fig3_consolidated_rank.png       전체 RRF ranking 막대 (Open/Closed 색깔 구분)
  fig4_domain_heatmap.png          도메인별 상위 10 모델 acc 히트맵
  fig5_model_size_frontier.png     Open Weights 모델 크기 대비 정확도 + Pareto frontier
  fig6_domain_open_boxplot.png     Open Weights 도메인별 정확도 분포
  fig7_open_top10_judge_variability.png
                                    Open Top-10 모델별 judge variability
  fig8_leave_one_judge_out_rank_change.png
                                    judge leave-one-out ranking stability
  fig9_judge_spearman_matrix.png   judge ranking Spearman agreement

기존 fig1~fig4와 analysis_summary.json은 기본 실행에서 덮어쓰지 않는다.
"""
import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rag_eval_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
PUBLISH = ROOT / "publish"
IMG_DIR = ROOT / "docs/images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

TEAL = "#83ced2"
CORAL = "#ed7969"
TEXT = "#1e293b"
MUTED = "#64748b"
LIGHT_TEAL = "#d4f2f3"
LIGHT_CORAL = "#fde0dc"
BOOTSTRAP_RESAMPLES = 10_000
RNG_SEED = 20260519

OPEN_CANDS = {
    "deepseek-r1_70b_nothink", "exaone3.5_32b", "gpt-oss_20b", "gpt-oss_120b",
    "lfm2_24b", "mistral-small_24b", "phi4_14b",
    "qwen3.5_9b-q4_K_M_nothink", "qwen3.5_9b-q8_0_nothink", "qwen3.5_27b-q8_0_nothink",
    "qwen3.5_122b-a10b-q4_K_M_nothink", "qwen3.5_122b-a10b-q4_K_M_think",
    "deepseek_deepseek-v3.2", "deepseek_deepseek-v4-flash", "deepseek_deepseek-v4-pro",
    "moonshotai_kimi-k2.5", "moonshotai_kimi-k2.6",
    "minimax_minimax-m2.5", "minimax_minimax-m2.7",
    "mistralai_mistral-small-2603", "nvidia_nemotron-3-nano-30b-a3b",
    "xiaomi_mimo-v2.5", "xiaomi_mimo-v2.5-pro",
    "z-ai_glm-4.7", "z-ai_glm-4.7-flash", "z-ai_glm-5", "z-ai_glm-5.1",
}
OPEN_JUDGES = ["gemma4_31b","nemotron-3-super-120b","qwen3-next-80b",
               "qwen3.5-27b-claude-distill","qwen3.5_122b-a10b-q4_K_M","qwen3.5_35b-a3b-q4_K_M",
               "qwen3.6_35b-a3b-q4_K_M","solar-open-100b","supergemma4-26b"]
CLOSED_JUDGES = ["claude-opus-4-7","claude-sonnet-4-6",
                 "gemini-3-flash-preview","gemini-3.1-flash-lite-preview","gemini-3.1-pro-preview",
                 "gpt-5.4","gpt-5.4-mini","gpt-5.4-nano","gpt-5.5"]

ALL_JUDGES = OPEN_JUDGES + CLOSED_JUDGES

PARAM_SIZE_B = {
    "deepseek-r1_70b_nothink": 70,
    "exaone3.5_32b": 32,
    "gpt-oss_20b": 20,
    "gpt-oss_120b": 120,
    "lfm2_24b": 24,
    "mistral-small_24b": 24,
    "phi4_14b": 14,
    "qwen3.5_9b-q4_K_M_nothink": 9,
    "qwen3.5_9b-q8_0_nothink": 9,
    "qwen3.5_27b-q8_0_nothink": 27,
    "qwen3.5_122b-a10b-q4_K_M_nothink": 122,
    "qwen3.5_122b-a10b-q4_K_M_think": 122,
    "moonshotai_kimi-k2.5": 1_000,
    "moonshotai_kimi-k2.6": 1_000,
    "mistralai_mistral-small-2603": 24,
    "nvidia_nemotron-3-nano-30b-a3b": 30,
}


def warn(msg):
    print(f"WARNING: {msg}")


def setup_style():
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "font.family": ["Pretendard", "Noto Sans KR", "AppleGothic", "DejaVu Sans", "sans-serif"],
        },
    )


def save_fig(fig, path, preserve_existing=False, dpi=150):
    if preserve_existing and path.exists():
        print(f"kept existing {path.name}")
        plt.close(fig)
        return False
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path.name}")
    return True


def read_parquet_safe(path):
    if not path.exists():
        warn(f"missing parquet: {path}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        warn(f"failed to read {path}: {exc}")
        return None


def load_inputs():
    cons = read_parquet_safe(PUBLISH / "consolidated.parquet")
    if cons is None:
        raise FileNotFoundError(PUBLISH / "consolidated.parquet")
    jd = {}
    for judge in ALL_JUDGES:
        frame = read_parquet_safe(PUBLISH / f"{judge}.parquet")
        if frame is not None:
            jd[judge] = frame
    missing = sorted(set(ALL_JUDGES) - set(jd))
    if missing:
        warn(f"skipping missing judge files: {', '.join(missing)}")
    return cons, jd


def get_ox(frame, cand, source_name):
    col = f"{cand}_ox"
    if frame is None or col not in frame.columns:
        warn(f"missing column {col} in {source_name}; skipped")
        return None
    return frame[col].eq("O").to_numpy(dtype=bool)


def wilson_ci(k, n, z=1.959963984540054):
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def acc_ci_from_bool(values):
    values = np.asarray(values, dtype=bool)
    n = int(values.size)
    k = int(values.sum())
    acc = k / n if n else np.nan
    lo, hi = wilson_ci(k, n)
    return acc, lo, hi, n, k


def fmt_ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def ci_overlap(a_lo, a_hi, b_lo, b_hi):
    return bool(max(a_lo, b_lo) <= min(a_hi, b_hi))


def judge_pool_accuracy(jd, judges, cand):
    arrs = []
    per_judge = []
    used = []
    for judge in judges:
        arr = get_ox(jd.get(judge), cand, judge)
        if arr is None:
            continue
        arrs.append(arr)
        per_judge.append(float(arr.mean()))
        used.append(judge)
    if not arrs:
        return np.nan, np.nan, np.nan, 0, []
    combined = np.concatenate(arrs)
    _acc, lo, hi, n, _k = acc_ci_from_bool(combined)
    return float(np.mean(per_judge)), lo, hi, n, used


def build_candidate_frame(cons, jd):
    all_cands = sorted([c[:-3] for c in cons.columns if c.endswith("_ox")])
    data = []
    for c in all_cands:
        cons_arr = get_ox(cons, c, "consolidated")
        if cons_arr is None:
            continue
        cons_acc, cons_lo, cons_hi, cons_n, cons_k = acc_ci_from_bool(cons_arr)
        open_acc, open_lo, open_hi, open_n, _open_used = judge_pool_accuracy(jd, OPEN_JUDGES, c)
        closed_acc, closed_lo, closed_hi, closed_n, _closed_used = judge_pool_accuracy(jd, CLOSED_JUDGES, c)
        group = "Open" if c in OPEN_CANDS else "Closed"
        data.append({
            "cand": c,
            "group": group,
            "open_acc": open_acc,
            "open_ci_low": open_lo,
            "open_ci_high": open_hi,
            "open_n": open_n,
            "closed_acc": closed_acc,
            "closed_ci_low": closed_lo,
            "closed_ci_high": closed_hi,
            "closed_n": closed_n,
            "cons_acc": cons_acc,
            "cons_ci_low": cons_lo,
            "cons_ci_high": cons_hi,
            "cons_n": cons_n,
            "cons_k": cons_k,
        })
    df = pd.DataFrame(data).sort_values(["cons_acc", "cand"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df, all_cands


def infer_param_b(cand):
    if cand in PARAM_SIZE_B:
        return float(PARAM_SIZE_B[cand])
    match = re.search(r"(\d+(?:\.\d+)?)b", cand, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return np.nan


def cand_family(name):
    n = name.lower()
    if "qwen" in n:
        return "qwen"
    if "gpt-oss" in n:
        return "gpt"
    if n.startswith("gpt-"):
        return "gpt"
    if "claude" in n:
        return "claude"
    if "gemini" in n:
        return "gemini"
    if "mistral" in n:
        return "mistral"
    if "kimi" in n or "moonshot" in n:
        return "kimi"
    if "deepseek" in n:
        return "deepseek"
    if "minimax" in n:
        return "minimax"
    if "glm" in n or "z-ai" in n:
        return "glm"
    if "gemma" in n:
        return "gemma"
    if "nemotron" in n or "nvidia" in n:
        return "nemotron"
    if "solar" in n:
        return "solar"
    if "exaone" in n:
        return "exaone"
    if "phi" in n:
        return "phi"
    if "lfm" in n:
        return "lfm"
    if "mimo" in n or "xiaomi" in n:
        return "mimo"
    return "other"


def judge_family(name):
    n = name.lower()
    if "qwen" in n:
        return "qwen"
    if "gpt" in n:
        return "gpt"
    if "claude" in n:
        return "claude"
    if "gemini" in n:
        return "gemini"
    if "gemma" in n:
        return "gemma"
    if "nemotron" in n:
        return "nemotron"
    if "solar" in n:
        return "solar"
    if "mistral" in n:
        return "mistral"
    if "llama" in n:
        return "llama"
    if "gpt-oss" in n:
        return "gpt-oss"
    return "other"


def bootstrap_top_gap(cons, df, n_resamples=BOOTSTRAP_RESAMPLES, seed=RNG_SEED):
    open_top = df[df["group"] == "Open"].iloc[0]
    closed_top = df[df["group"] == "Closed"].iloc[0]
    open_arr = get_ox(cons, open_top["cand"], "consolidated")
    closed_arr = get_ox(cons, closed_top["cand"], "consolidated")
    n = min(len(open_arr), len(closed_arr))
    open_arr = open_arr[:n].astype(float)
    closed_arr = closed_arr[:n].astype(float)
    obs = float(closed_arr.mean() - open_arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = closed_arr[idx].mean(axis=1) - open_arr[idx].mean(axis=1)
    p_left = (np.count_nonzero(diffs <= 0) + 1) / (n_resamples + 1)
    p_right = (np.count_nonzero(diffs >= 0) + 1) / (n_resamples + 1)
    p_value = float(min(1.0, 2 * min(p_left, p_right)))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {
        "open_top": open_top["cand"],
        "closed_top": closed_top["cand"],
        "open_acc": float(open_top["cons_acc"]),
        "closed_acc": float(closed_top["cons_acc"]),
        "observed_diff_closed_minus_open": obs,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "significant_95": bool(p_value < 0.05 and not (ci_low <= 0 <= ci_high)),
        "n_resamples": n_resamples,
    }


def plot_existing_figures(cons, df, all_cands):
    # === Fig 1: open-pool vs closed-pool 산점도 ===
    fig, ax = plt.subplots(figsize=(11, 9))
    for g, color, marker in [("Open", TEAL, "o"), ("Closed", CORAL, "s")]:
        sub = df[df["group"] == g]
        ax.scatter(sub["closed_acc"], sub["open_acc"], c=color, marker=marker, s=80, alpha=0.85,
                   edgecolors="black", linewidth=0.5, label=f"{g} Weights ({len(sub)})")
    lims = [0.3, 0.85]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
    z = np.polyfit(df["closed_acc"], df["open_acc"], 1)
    xs = np.linspace(*lims, 100)
    ax.plot(xs, z[0] * xs + z[1], "-", color="#1a5c5f", alpha=0.75,
            label=f"trend: y = {z[0]:.3f}x + {z[1]:.3f}")
    annotate_cands = list(df.nlargest(8, "cons_acc")["cand"]) + ["gpt-oss_120b", "moonshotai_kimi-k2.5", "qwen3.5_122b-a10b-q4_K_M_think"]
    for _, row in df.iterrows():
        if row["cand"] in annotate_cands:
            ax.annotate(row["cand"], (row["closed_acc"], row["open_acc"]),
                        fontsize=8, alpha=0.85, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("Closed Weights judge pool — mean accuracy", fontsize=11)
    ax.set_ylabel("Open Weights judge pool — mean accuracy", fontsize=11)
    ax.set_title("Generation model accuracy: Open-weights pool vs Closed-weights pool\n(each point = one of 46 generation models)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims); ax.set_ylim(lims)
    save_fig(fig, IMG_DIR / "fig1_open_vs_closed_scatter.png", preserve_existing=True)

    # === Fig 2: GPT-5.4-pro 기준 alternatives ===
    benchmarks = ["gpt-5.4-pro", "gpt-5.4", "gpt-5.4-mini", "gemini-3.1-pro-preview", "claude-opus-4-7"]
    open_top = df[df["group"] == "Open"].sort_values(["cons_acc", "cand"], ascending=[False, True]).head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(open_top))
    ax.barh(y_pos, open_top["cons_acc"], color=TEAL, alpha=0.85, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(open_top["cand"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Consolidated accuracy (18-judge majority)", fontsize=11)
    ax.set_title("Open-weights alternatives to Closed-weights models — Top 15", fontsize=12)

    colors = [CORAL, "#8a3028", "#bcbd22", "#9467bd", "#8c564b"]
    for bm, color in zip(benchmarks, colors):
        bm_match = df[df["cand"] == bm]
        if bm_match.empty:
            warn(f"benchmark missing from candidate frame: {bm}")
            continue
        bm_acc = bm_match["cons_acc"].iloc[0]
        ax.axvline(bm_acc, color=color, linestyle="--", alpha=0.8, label=f"{bm} = {bm_acc:.3f}")

    gpt54pro_match = df[df["cand"] == "gpt-5.4-pro"]
    if not gpt54pro_match.empty:
        gpt54pro_acc = gpt54pro_match["cons_acc"].iloc[0]
        for i, (_, row) in enumerate(open_top.iterrows()):
            gap = (row["cons_acc"] - gpt54pro_acc) * 100
            ax.text(row["cons_acc"] + 0.005, i, f"Δ{gap:+.1f}pp", va="center", fontsize=8, color="gray")

    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0.5, 0.85)
    ax.grid(True, axis="x", alpha=0.3)
    save_fig(fig, IMG_DIR / "fig2_alternative_to_gpt54.png", preserve_existing=True)

    # === Fig 3: 전체 ranking 막대 (cons_acc) ===
    fig, ax = plt.subplots(figsize=(11, 14))
    y_pos = np.arange(len(df))
    colors = [TEAL if g == "Open" else CORAL for g in df["group"]]
    ax.barh(y_pos, df["cons_acc"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["cand"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Consolidated accuracy", fontsize=11)
    ax.set_title("All generation models — consolidated ranking\nTeal = Open Weights, Coral = Closed Weights", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0.3, 0.85)
    save_fig(fig, IMG_DIR / "fig3_consolidated_rank.png", preserve_existing=True)

    # === Fig 4: 도메인별 heatmap (Top 15 cand) ===
    domain_acc = domain_accuracy(cons, all_cands)
    top15 = df.head(15)["cand"].tolist()
    domains = [d for d in ["finance", "public", "medical", "law", "commerce"] if d in domain_acc.columns]
    mat = np.array([[domain_acc.loc[c, d] for d in domains] for c in top15])

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(domains))); ax.set_xticklabels(domains, fontsize=10)
    ax.set_yticks(range(len(top15))); ax.set_yticklabels(top15, fontsize=9)
    for i, c in enumerate(top15):
        g = "Open" if c in OPEN_CANDS else "Closed"
        color = TEAL if g == "Open" else CORAL
        ax.add_patch(plt.Rectangle((-0.7, i - 0.5), 0.2, 1, facecolor=color, clip_on=False, edgecolor="none"))
    for i in range(len(top15)):
        for j in range(len(domains)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, label="accuracy")
    ax.set_title("Domain-level accuracy — Top 15 generation models", fontsize=12)
    save_fig(fig, IMG_DIR / "fig4_domain_heatmap.png", preserve_existing=True)
    return z


def domain_accuracy(cons, candidates):
    records = {}
    domains = list(cons["domain"].dropna().unique())
    for cand in candidates:
        col = f"{cand}_ox"
        if col not in cons.columns:
            warn(f"missing column {col} in consolidated; skipped for domain accuracy")
            continue
        records[cand] = {}
        for dom in domains:
            mask = cons["domain"] == dom
            records[cand][dom] = float(cons.loc[mask, col].eq("O").mean())
    return pd.DataFrame.from_dict(records, orient="index")


def plot_model_size_frontier(df):
    sub = df[df["group"] == "Open"].copy()
    sub["params_b"] = sub["cand"].map(infer_param_b)
    sub = sub.dropna(subset=["params_b"]).sort_values("params_b")
    if sub.empty:
        warn("fig5 skipped: no Open Weights candidates with known parameter sizes")
        return None
    sub["acc_per_b"] = sub["cons_acc"] / sub["params_b"]
    frontier_rows = []
    best_acc = -np.inf
    for _, row in sub.iterrows():
        if row["cons_acc"] > best_acc:
            frontier_rows.append(row)
            best_acc = row["cons_acc"]
    frontier = pd.DataFrame(frontier_rows)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(sub["params_b"], sub["cons_acc"], s=95, color=TEAL, edgecolor="black", linewidth=0.6, alpha=0.88)
    ax.plot(frontier["params_b"], frontier["cons_acc"], color=CORAL, linewidth=2.5, marker="o",
            label="Pareto frontier")
    for _, row in sub.sort_values("cons_acc", ascending=False).head(8).iterrows():
        ax.annotate(row["cand"], (row["params_b"], row["cons_acc"]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (B parameters, log scale)")
    ax.set_ylabel("Consolidated accuracy")
    ax.set_title("Open Weights model size vs accuracy with Pareto frontier")
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.legend()
    save_fig(fig, IMG_DIR / "fig5_model_size_frontier.png")
    return sub.sort_values("acc_per_b", ascending=False)


def plot_domain_boxplot(cons, df):
    records = []
    for cand in df[df["group"] == "Open"]["cand"]:
        arr = get_ox(cons, cand, "consolidated")
        if arr is None:
            continue
        for dom in cons["domain"].dropna().unique():
            mask = cons["domain"] == dom
            records.append({
                "cand": cand,
                "domain": dom,
                "accuracy": float(cons.loc[mask, f"{cand}_ox"].eq("O").mean()),
            })
    dom_df = pd.DataFrame(records)
    if dom_df.empty:
        warn("fig6 skipped: no domain-level Open Weights records")
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    order = dom_df.groupby("domain")["accuracy"].median().sort_values(ascending=False).index.tolist()
    sns.boxplot(data=dom_df, x="domain", y="accuracy", order=order, color=LIGHT_TEAL, linewidth=1.2, ax=ax)
    sns.stripplot(data=dom_df, x="domain", y="accuracy", order=order, color=TEAL, edgecolor="black",
                  linewidth=0.3, alpha=0.65, size=4, ax=ax)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Accuracy across Open Weights candidates")
    ax.set_title("Domain-level accuracy distribution across Open Weights candidates")
    ax.set_ylim(0.25, 0.95)
    save_fig(fig, IMG_DIR / "fig6_domain_open_boxplot.png")
    variance = dom_df.groupby("domain")["accuracy"].agg(["mean", "std", "min", "max"]).sort_values("std", ascending=False)
    return variance


def plot_judge_variability(jd, df):
    top10 = df[df["group"] == "Open"].head(10)["cand"].tolist()
    records = []
    for cand in top10:
        for judge in ALL_JUDGES:
            arr = get_ox(jd.get(judge), cand, judge)
            if arr is None:
                continue
            records.append({
                "cand": cand,
                "judge": judge,
                "judge_group": "Open judge" if judge in OPEN_JUDGES else "Closed judge",
                "accuracy": float(arr.mean()),
            })
    var_df = pd.DataFrame(records)
    if var_df.empty:
        warn("fig7 skipped: no judge variability records")
        return None
    fig, ax = plt.subplots(figsize=(12, 7))
    order = top10[::-1]
    sns.boxplot(data=var_df, y="cand", x="accuracy", order=order, color=LIGHT_TEAL, linewidth=1.2, ax=ax)
    sns.stripplot(data=var_df, y="cand", x="accuracy", order=order, hue="judge_group",
                  palette={"Open judge": TEAL, "Closed judge": CORAL}, dodge=True,
                  edgecolor="black", linewidth=0.35, alpha=0.78, size=5, ax=ax)
    ax.set_xlabel("Per-judge accuracy")
    ax.set_ylabel("")
    ax.set_title("Judge variability for Open Weights Top-10 candidates")
    ax.set_xlim(0.35, 0.9)
    ax.legend(title="", loc="lower right")
    save_fig(fig, IMG_DIR / "fig7_open_top10_judge_variability.png")
    return var_df.groupby("cand")["accuracy"].agg(["mean", "std", "min", "max"]).sort_values("std", ascending=False)


def majority_accuracy_without_judge(jd, candidates, removed_judge):
    rows = []
    remaining = [j for j in ALL_JUDGES if j != removed_judge and j in jd]
    for cand in candidates:
        votes = []
        for judge in remaining:
            arr = get_ox(jd.get(judge), cand, judge)
            if arr is not None:
                votes.append(arr.astype(int))
        if not votes:
            continue
        vote_mat = np.vstack(votes)
        majority = vote_mat.mean(axis=0) > 0.5
        rows.append({"cand": cand, "accuracy": float(majority.mean())})
    out = pd.DataFrame(rows).sort_values(["accuracy", "cand"], ascending=[False, True]).reset_index(drop=True)
    out["rank_removed"] = out.index + 1
    return out


def plot_leave_one_out(cons, jd, df):
    open_cands = df[df["group"] == "Open"]["cand"].tolist()
    base_rank = df[df["group"] == "Open"].set_index("cand")["rank"].rank(method="dense").astype(int)
    records = []
    for removed in [j for j in ALL_JUDGES if j in jd]:
        loo = majority_accuracy_without_judge(jd, open_cands, removed)
        for _, row in loo.iterrows():
            records.append({
                "cand": row["cand"],
                "removed_judge": removed,
                "rank_change": int(row["rank_removed"] - base_rank.loc[row["cand"]]),
            })
    loo_df = pd.DataFrame(records)
    if loo_df.empty:
        warn("fig8 skipped: no leave-one-out ranking records")
        return None
    heat = loo_df.pivot(index="cand", columns="removed_judge", values="rank_change")
    ordered_models = df[df["group"] == "Open"]["cand"].tolist()
    heat = heat.reindex(ordered_models)
    fig, ax = plt.subplots(figsize=(14, 10))
    max_abs = int(np.nanmax(np.abs(heat.to_numpy()))) if not heat.empty else 1
    sns.heatmap(heat, cmap="vlag", center=0, vmin=-max_abs, vmax=max_abs, annot=True, fmt=".0f",
                linewidths=0.4, linecolor="#e5e7eb", cbar_kws={"label": "rank change (+ worse, - better)"}, ax=ax)
    ax.set_xlabel("Removed judge")
    ax.set_ylabel("Open Weights candidate")
    ax.set_title("Ranking stability: rank change after removing one judge")
    ax.tick_params(axis="x", labelrotation=45)
    save_fig(fig, IMG_DIR / "fig8_leave_one_judge_out_rank_change.png")
    return loo_df


def plot_judge_agreement(jd, df):
    open_cands = df[df["group"] == "Open"]["cand"].tolist()
    score_cols = {}
    for judge in [j for j in ALL_JUDGES if j in jd]:
        scores = {}
        for cand in open_cands:
            arr = get_ox(jd.get(judge), cand, judge)
            if arr is not None:
                scores[cand] = float(arr.mean())
        if scores:
            score_cols[judge] = pd.Series(scores)
    scores_df = pd.DataFrame(score_cols)
    if scores_df.shape[1] < 2:
        warn("fig9 skipped: fewer than two judges with usable scores")
        return None
    corr = scores_df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="vlag", vmin=-1, vmax=1, center=0, linewidths=0.3, linecolor="#e5e7eb",
                cbar_kws={"label": "Spearman correlation"}, ax=ax)
    ax.set_title("Judge agreement matrix on Open Weights candidate rankings")
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    save_fig(fig, IMG_DIR / "fig9_judge_spearman_matrix.png")
    return corr


def top_table_with_ci(df, group, n=10, benchmark=None):
    sub = df[df["group"] == group].head(n).copy()
    if benchmark is not None and not df[df["cand"] == benchmark].empty:
        bm = df[df["cand"] == benchmark].iloc[0]
        sub["gap_vs_benchmark_pp"] = (sub["cons_acc"] - bm["cons_acc"]) * 100
        sub["ci_overlap_benchmark"] = sub.apply(
            lambda r: ci_overlap(r["cons_ci_low"], r["cons_ci_high"], bm["cons_ci_low"], bm["cons_ci_high"]),
            axis=1,
        )
    return sub


def judge_leniency(jd, candidates):
    rows = []
    for judge in [j for j in ALL_JUDGES if j in jd]:
        vals = []
        for cand in candidates:
            arr = get_ox(jd.get(judge), cand, judge)
            if arr is not None:
                vals.append(arr)
        if not vals:
            continue
        combined = np.concatenate(vals)
        rows.append({
            "judge": judge,
            "family": judge_family(judge),
            "group": "Open" if judge in OPEN_JUDGES else "Closed",
            "mean_acc_assigned": float(combined.mean()),
        })
    judge_df = pd.DataFrame(rows).sort_values("mean_acc_assigned")
    family_df = judge_df.groupby("family").agg(
        mean_acc_assigned=("mean_acc_assigned", "mean"),
        n_judges=("judge", "count"),
    ).sort_values("mean_acc_assigned")
    return judge_df, family_df


def self_bias_rank_delta(jd, df, candidates):
    cand_fams = pd.Series({cand: cand_family(cand) for cand in candidates})
    judge_fams = pd.Series({judge: judge_family(judge) for judge in jd})
    rows = []
    for fam in sorted(set(cand_fams) & set(judge_fams)):
        fam_judges = judge_fams[judge_fams == fam].index.tolist()
        other_judges = judge_fams[judge_fams != fam].index.tolist()
        fam_cands = cand_fams[cand_fams == fam].index.tolist()
        if not fam_judges or not other_judges or not fam_cands:
            continue

        def score_with(judges):
            scores = {}
            for cand in candidates:
                vals = []
                for judge in judges:
                    arr = get_ox(jd.get(judge), cand, judge)
                    if arr is not None:
                        vals.append(float(arr.mean()))
                if vals:
                    scores[cand] = float(np.mean(vals))
            return pd.Series(scores)

        own_scores = score_with(fam_judges)
        other_scores = score_with(other_judges)
        common = own_scores.index.intersection(other_scores.index)
        own_rank = own_scores.loc[common].rank(ascending=False, method="min")
        other_rank = other_scores.loc[common].rank(ascending=False, method="min")
        fam_common = [c for c in fam_cands if c in common]
        if not fam_common:
            continue
        own_mean_rank = float(own_rank.loc[fam_common].mean())
        other_mean_rank = float(other_rank.loc[fam_common].mean())
        rows.append({
            "family": fam,
            "n_family_judges": len(fam_judges),
            "n_family_candidates": len(fam_common),
            "own_judge_mean_rank": own_mean_rank,
            "other_judge_mean_rank": other_mean_rank,
            "rank_delta_other_minus_own": other_mean_rank - own_mean_rank,
        })
    return pd.DataFrame(rows).sort_values("rank_delta_other_minus_own", ascending=False)


def domain_weighted_ranking(cons, df, candidates):
    domain_counts = cons["domain"].value_counts()
    top_domains = domain_counts.head(3).index.tolist()
    mask = cons["domain"].isin(top_domains)
    rows = []
    for cand in candidates:
        col = f"{cand}_ox"
        if col not in cons.columns:
            continue
        rows.append({
            "cand": cand,
            "group": "Open" if cand in OPEN_CANDS else "Closed",
            "top3_domain_acc": float(cons.loc[mask, col].eq("O").mean()),
            "overall_rank": int(df.set_index("cand").loc[cand, "rank"]),
        })
    out = pd.DataFrame(rows).sort_values(["top3_domain_acc", "cand"], ascending=[False, True]).reset_index(drop=True)
    out["top3_domain_rank"] = out.index + 1
    return top_domains, domain_counts.loc[top_domains], out


def hardest_domains(cons, candidates):
    domain_df = domain_accuracy(cons, candidates)
    return domain_df.mean(axis=0).sort_values()


def family_accuracy_stats(df):
    fam_df = df.copy()
    fam_df["family"] = fam_df["cand"].map(cand_family)
    return fam_df.groupby(["group", "family"]).agg(
        n=("cand", "count"),
        mean_acc=("cons_acc", "mean"),
        max_acc=("cons_acc", "max"),
        std_acc=("cons_acc", "std"),
    ).reset_index().sort_values(["group", "mean_acc"], ascending=[True, False])


def estimate_token_cost(cons, candidates):
    answer_cols = [f"{c}_answer" for c in candidates if f"{c}_answer" in cons.columns]
    question_tokens = cons["question"].fillna("").str.len().mean() / 3.2
    target_tokens = cons["target_answer"].fillna("").str.len().mean() / 3.2
    answer_tokens = cons[answer_cols].fillna("").astype(str).apply(lambda s: s.str.len() / 3.2).stack().mean()
    per_generation_tokens = float(question_tokens + target_tokens + answer_tokens)
    benchmark_generation_tokens_m = per_generation_tokens * len(cons) * len(candidates) / 1_000_000
    benchmark_judge_tokens_m = (question_tokens + target_tokens + answer_tokens) * len(cons) * len(candidates) * len(ALL_JUDGES) / 1_000_000
    return {
        "avg_question_tokens": float(question_tokens),
        "avg_target_tokens": float(target_tokens),
        "avg_answer_tokens": float(answer_tokens),
        "per_generation_tokens": per_generation_tokens,
        "benchmark_generation_tokens_m": float(benchmark_generation_tokens_m),
        "benchmark_judge_tokens_m": float(benchmark_judge_tokens_m),
    }


def build_extended_summary(cons, jd, df, all_cands, trend):
    bootstrap = bootstrap_top_gap(cons, df)
    gpt54pro_acc = float(df[df["cand"] == "gpt-5.4-pro"]["cons_acc"].iloc[0]) if not df[df["cand"] == "gpt-5.4-pro"].empty else np.nan
    gpt54mini_acc = float(df[df["cand"] == "gpt-5.4-mini"]["cons_acc"].iloc[0]) if not df[df["cand"] == "gpt-5.4-mini"].empty else np.nan
    open_top_ci = top_table_with_ci(df, "Open", 15, benchmark="gpt-5.4-pro")
    closed_top_ci = top_table_with_ci(df, "Closed", 10)
    judge_df, judge_family_df = judge_leniency(jd, all_cands)
    bias_rank = self_bias_rank_delta(jd, df, all_cands)
    top_domains, top_domain_counts, domain_weighted = domain_weighted_ranking(cons, df, all_cands)
    hard = hardest_domains(cons, all_cands)
    fam_stats = family_accuracy_stats(df)
    token_cost = estimate_token_cost(cons, all_cands)

    return {
        "n_cands": len(df),
        "n_open": int((df["group"] == "Open").sum()),
        "n_closed": int((df["group"] == "Closed").sum()),
        "open_avg": float(df[df["group"] == "Open"]["cons_acc"].mean()),
        "closed_avg": float(df[df["group"] == "Closed"]["cons_acc"].mean()),
        "trend_slope": float(trend[0]),
        "trend_intercept": float(trend[1]),
        "correlation_open_vs_closed_pool": float(np.corrcoef(df["closed_acc"], df["open_acc"])[0, 1]),
        "gpt54pro_cons_acc": gpt54pro_acc,
        "gpt54mini_cons_acc": gpt54mini_acc,
        "bootstrap_top_gap": bootstrap,
        "open_top_ci": open_top_ci,
        "closed_top_ci": closed_top_ci,
        "judge_leniency": judge_df,
        "judge_family_leniency": judge_family_df,
        "self_bias_rank_delta": bias_rank,
        "top3_domains": top_domains,
        "top3_domain_counts": top_domain_counts,
        "domain_weighted_ranking": domain_weighted,
        "hardest_domains": hard,
        "family_accuracy_stats": fam_stats,
        "token_cost": token_cost,
    }


def print_summary(summary, efficiency, domain_variance, judge_variability, loo_df, corr):
    boot = summary["bootstrap_top_gap"]
    print("=== statistical rigor ===")
    print(
        f"Top Open {boot['open_top']}={boot['open_acc']:.3f}; "
        f"Top Closed {boot['closed_top']}={boot['closed_acc']:.3f}; "
        f"closed-open diff={boot['observed_diff_closed_minus_open']:.3f}; "
        f"bootstrap p={boot['p_value']:.4f}; "
        f"95% bootstrap CI={fmt_ci(boot['ci_low'], boot['ci_high'])}; "
        f"significant={boot['significant_95']}"
    )
    for _, row in summary["open_top_ci"].head(5).iterrows():
        noise = "CI-overlap" if row.get("ci_overlap_benchmark", False) else "separated"
        print(f"Open top CI: {row['cand']} acc={row['cons_acc']:.3f} CI={fmt_ci(row['cons_ci_low'], row['cons_ci_high'])} vs gpt-5.4-pro={noise}")
    for _, row in summary["closed_top_ci"].head(3).iterrows():
        print(f"Closed top CI: {row['cand']} acc={row['cons_acc']:.3f} CI={fmt_ci(row['cons_ci_low'], row['cons_ci_high'])}")

    if efficiency is not None and not efficiency.empty:
        top_eff = efficiency.iloc[0]
        print(f"efficiency frontier winner: {top_eff['cand']} acc/B={top_eff['acc_per_b']:.4f}, acc={top_eff['cons_acc']:.3f}, params={top_eff['params_b']:.1f}B")
    if domain_variance is not None and not domain_variance.empty:
        print("highest variance domains: " + ", ".join(f"{idx}(std={row['std']:.3f})" for idx, row in domain_variance.head(3).iterrows()))
    if judge_variability is not None and not judge_variability.empty:
        print("highest judge variability top10: " + ", ".join(f"{idx}(std={row['std']:.3f})" for idx, row in judge_variability.head(3).iterrows()))
    if loo_df is not None and not loo_df.empty:
        print(f"leave-one-out max abs rank change: {int(loo_df['rank_change'].abs().max())}")
    if corr is not None and not corr.empty:
        vals = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack()
        print(f"judge agreement Spearman median={vals.median():.3f}, min={vals.min():.3f}, max={vals.max():.3f}")


def main():
    setup_style()
    cons, jd = load_inputs()
    df, all_cands = build_candidate_frame(cons, jd)

    trend = plot_existing_figures(cons, df, all_cands)
    efficiency = plot_model_size_frontier(df)
    domain_variance = plot_domain_boxplot(cons, df)
    judge_variability = plot_judge_variability(jd, df)
    loo_df = plot_leave_one_out(cons, jd, df)
    corr = plot_judge_agreement(jd, df)
    summary = build_extended_summary(cons, jd, df, all_cands, trend)

    # Keep docs/analysis_summary.json untouched unless explicitly requested.
    if os.environ.get("WRITE_ANALYSIS_SUMMARY") == "1":
        with open(ROOT / "docs/analysis_summary.json", "w") as f:
            json.dump({
                "n_cands": summary["n_cands"],
                "n_open": summary["n_open"],
                "n_closed": summary["n_closed"],
                "open_avg": summary["open_avg"],
                "closed_avg": summary["closed_avg"],
                "bootstrap_top_gap": summary["bootstrap_top_gap"],
            }, f, ensure_ascii=False, indent=2)
        print("saved analysis_summary.json")
    else:
        print("kept existing analysis_summary.json")

    print_summary(summary, efficiency, domain_variance, judge_variability, loo_df, corr)


if __name__ == "__main__":
    main()
