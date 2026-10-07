"""
run_report_full23.py
====================
读取 run_evaluation_full23.py 生成的 evaluation_results_full23.csv，
汇总 23 个指标，生成表格和图，并筛选 MRGEOA/m2geo 表现好的指标。

默认输入：outputs/scores/evaluation_results_full23.csv
默认输出：outputs/tables_full23/ 与 outputs/figures_full23/

运行：
    python run_report_full23.py --target m2geo
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR
try:
    from config import METHOD_LIST
except Exception:
    METHOD_LIST = []


JUDGE_METRICS = [
    "citation_prominence", "answer_dominance", "visibility_overall",
    "relevance", "influence", "uniqueness", "diversity",
    "click_likelihood", "subjective_position", "subjective_volume",
    "attribution_accuracy", "faithfulness", "evidence_precision", "evidence_recall",
    "key_point_coverage", "semantic_contribution", "key_point_recall",
    "key_point_contradiction",
    "clarity", "insight", "coherence", "structure_quality", "readability",
]
SUMMARY_METRICS = ["objective_score", "subjective_score", "faithfulness_score", "utility_score", "overall_score"]
ALL_METRICS = JUDGE_METRICS + SUMMARY_METRICS
LOWER_IS_BETTER = {"key_point_contradiction"}
MAIN_METRICS = [
    "visibility_overall", "relevance", "influence", "click_likelihood", "faithfulness",
    "attribution_accuracy", "key_point_coverage", "semantic_contribution",
    "structure_quality", "overall_score",
]

METHOD_LABELS = {
    "uniqueness": "Uniqueness", "simplification": "Simplification", "authority": "Authority",
    "fluency": "Fluency", "terminology": "Terminology", "reputation": "Reputation",
    "citation": "Citation", "statistics": "Statistics", "autogeo": "AutoGEO",
    "intent_geo": "Intent-GEO", "multimodal": "Multimodal", "multi_agent": "Multi-Agent",
    "rag_based": "RAG-based", "m2geo": "MRGEOA", "mrgeoa": "MRGEOA", "MRGEOA": "MRGEOA",
}

METRIC_LABELS = {
    "citation_prominence": "Citation Prominence", "answer_dominance": "Answer Dominance",
    "visibility_overall": "Visibility Overall", "relevance": "Relevance", "influence": "Influence",
    "uniqueness": "Uniqueness", "diversity": "Diversity", "click_likelihood": "Click Likelihood",
    "subjective_position": "Subjective Position", "subjective_volume": "Subjective Volume",
    "attribution_accuracy": "Attribution Accuracy", "faithfulness": "Faithfulness",
    "evidence_precision": "Evidence Precision", "evidence_recall": "Evidence Recall",
    "key_point_coverage": "Key Point Coverage", "semantic_contribution": "Semantic Contribution",
    "key_point_recall": "Key Point Recall", "key_point_contradiction": "Key Point Contradiction",
    "clarity": "Clarity", "insight": "Insight", "coherence": "Coherence",
    "structure_quality": "Structure Quality", "readability": "Readability",
    "objective_score": "Objective Score", "subjective_score": "Subjective Score",
    "faithfulness_score": "Faithfulness Score", "utility_score": "Utility Score", "overall_score": "Overall Score",
}

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def ensure_dirs():
    tables_dir = OUTPUT_DIR / "tables_full23"
    figures_dir = OUTPUT_DIR / "figures_full23"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def load_scores(input_csv: Path):
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到评分文件：{input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    for col in ALL_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    available = [m for m in ALL_METRICS if m in df.columns]
    if not available:
        raise ValueError("评分文件中没有可用指标列。")
    method_mean = df.groupby("method")[available].mean(numeric_only=True).reset_index()
    method_std = df.groupby("method")[available].std(numeric_only=True).reset_index()
    order = [m for m in METHOD_LIST if m in method_mean["method"].values]
    others = [m for m in method_mean["method"].values if m not in order]
    final_order = order + others
    if final_order:
        method_mean = method_mean.set_index("method").reindex(final_order).reset_index()
        method_std = method_std.set_index("method").reindex(final_order).reset_index()
    method_mean["label"] = method_mean["method"].apply(method_label)
    method_std["label"] = method_std["method"].apply(method_label)
    return df, method_mean, method_std, available


def save_method_mean_table(method_mean: pd.DataFrame, available: List[str], tables_dir: Path):
    out = method_mean[["method", "label"] + available].copy()
    out.to_csv(tables_dir / "method_mean_scores_full23.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 方法平均分表：{tables_dir / 'method_mean_scores_full23.csv'}")


def rank_methods_for_metric(method_mean: pd.DataFrame, metric: str) -> pd.DataFrame:
    ascending = metric in LOWER_IS_BETTER
    sub = method_mean[["method", "label", metric]].dropna().copy()
    sub = sub.sort_values(metric, ascending=ascending).reset_index(drop=True)
    sub["rank"] = np.arange(1, len(sub) + 1)
    return sub


def compute_target_win_rate(df: pd.DataFrame, target: str, metric: str) -> float:
    if target not in df["method"].values or metric not in df.columns:
        return 0.0
    wins = 0
    total = 0
    for _, group in df.groupby("sample_id"):
        target_rows = group[group["method"] == target]
        other_rows = group[group["method"] != target]
        if target_rows.empty or other_rows.empty:
            continue
        try:
            target_val = float(target_rows[metric].iloc[0])
        except Exception:
            continue
        other_vals = pd.to_numeric(other_rows[metric], errors="coerce").dropna()
        if other_vals.empty:
            continue
        total += 1
        if metric in LOWER_IS_BETTER:
            wins += int(target_val <= other_vals.min())
        else:
            wins += int(target_val >= other_vals.max())
    return round(wins / total, 4) if total > 0 else 0.0


def save_advantage_tables(df: pd.DataFrame, method_mean: pd.DataFrame, available: List[str], target: str, tables_dir: Path):
    rows = []
    if target not in method_mean["method"].values:
        raise ValueError(f"目标方法 {target} 不在评分结果中。现有方法：{method_mean['method'].tolist()}")
    for metric in available:
        ranks = rank_methods_for_metric(method_mean, metric)
        target_row = ranks[ranks["method"] == target]
        if target_row.empty:
            continue
        target_rank = int(target_row["rank"].iloc[0])
        target_mean = float(target_row[metric].iloc[0])
        best_method = str(ranks.iloc[0]["method"])
        best_mean = float(ranks.iloc[0][metric])
        if len(ranks) >= 2:
            if target_rank == 1:
                second_mean = float(ranks.iloc[1][metric])
                gap = target_mean - second_mean
                if metric in LOWER_IS_BETTER:
                    gap = second_mean - target_mean
            else:
                gap = target_mean - best_mean
                if metric in LOWER_IS_BETTER:
                    gap = best_mean - target_mean
        else:
            gap = 0.0
        win_rate = compute_target_win_rate(df, target, metric)
        if target_rank == 1 and gap >= 0.10 and win_rate >= 0.45:
            recommendation = "优先保留"
        elif target_rank == 1 and gap >= 0.03:
            recommendation = "可以保留"
        elif target_rank <= 2:
            recommendation = "辅助分析"
        else:
            recommendation = "不建议主写"
        rows.append({
            "metric": metric, "metric_label": metric_label(metric),
            "target_method": target, "target_label": method_label(target),
            "target_mean": round(target_mean, 4), "rank": target_rank,
            "best_method": best_method, "best_label": method_label(best_method),
            "best_mean": round(best_mean, 4), "gap_to_second_or_best": round(gap, 4),
            "target_win_rate": win_rate, "recommendation": recommendation,
        })
    adv = pd.DataFrame(rows)
    adv = adv.sort_values(by=["rank", "gap_to_second_or_best", "target_win_rate"], ascending=[True, False, False])
    adv.to_csv(tables_dir / "mrgeoa_advantage_metrics_full23.csv", index=False, encoding="utf-8-sig")
    recommended = adv[adv["recommendation"].isin(["优先保留", "可以保留"])].copy()
    recommended.to_csv(tables_dir / "metric_recommendation_table_full23.csv", index=False, encoding="utf-8-sig")
    print(f"✅ MRGEOA优势指标表：{tables_dir / 'mrgeoa_advantage_metrics_full23.csv'}")
    print(f"✅ 推荐保留指标表：{tables_dir / 'metric_recommendation_table_full23.csv'}")
    return adv, recommended


def plot_main_metrics_bar(method_mean: pd.DataFrame, available: List[str], figures_dir: Path):
    metrics = [m for m in MAIN_METRICS if m in available] or available[:10]
    plot_df = method_mean[["label"] + metrics].copy()
    x = np.arange(len(metrics))
    n_methods = len(plot_df)
    width = 0.8 / max(n_methods, 1)
    fig, ax = plt.subplots(figsize=(max(14, len(metrics) * 1.5), 6))
    for i, (_, row) in enumerate(plot_df.iterrows()):
        vals = [row[m] for m in metrics]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=row["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([metric_label(m) for m in metrics], rotation=25, ha="right")
    ax.set_ylabel("Mean Score (0-5)")
    ax.set_ylim(0, 5.3)
    ax.set_title("Main Metrics Comparison across GEO Methods", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig1_main_metrics_bar.png", bbox_inches="tight")
    plt.close()
    print(f"✅ 图1：{figures_dir / 'fig1_main_metrics_bar.png'}")


def plot_heatmap(method_mean: pd.DataFrame, available: List[str], figures_dir: Path):
    metrics = [m for m in JUDGE_METRICS if m in available]
    if not metrics:
        return
    mat = method_mean.set_index("label")[metrics].copy()
    fig, ax = plt.subplots(figsize=(max(14, len(metrics) * 0.65), max(5, len(mat) * 0.45)))
    im = ax.imshow(mat.values, aspect="auto", vmin=0, vmax=5)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([metric_label(m) for m in metrics], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Full 23-Metric Heatmap", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score (0-5)")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig2_full23_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"✅ 图2：{figures_dir / 'fig2_full23_heatmap.png'}")


def plot_target_advantage(adv: pd.DataFrame, figures_dir: Path):
    if adv.empty:
        return
    top = adv.sort_values("gap_to_second_or_best", ascending=False).head(12)
    top = top.sort_values("gap_to_second_or_best", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.45)))
    ax.barh(top["metric_label"], top["gap_to_second_or_best"])
    ax.axvline(0, linewidth=1)
    for y, v in enumerate(top["gap_to_second_or_best"]):
        ax.text(v, y, f" {v:+.3f}", va="center", fontsize=9)
    ax.set_xlabel("Target Advantage Gap")
    ax.set_title("MRGEOA Advantage Metrics", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_mrgeoa_advantage.png", bbox_inches="tight")
    plt.close()
    print(f"✅ 图3：{figures_dir / 'fig3_mrgeoa_advantage.png'}")


def plot_overall_ranking(method_mean: pd.DataFrame, available: List[str], figures_dir: Path):
    if "overall_score" not in available:
        return
    df = method_mean[["label", "overall_score"]].dropna().sort_values("overall_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.45)))
    ax.barh(df["label"], df["overall_score"])
    ax.set_xlabel("Overall Score")
    ax.set_xlim(0, 5.3)
    for y, v in enumerate(df["overall_score"]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=9)
    ax.set_title("Overall Score Ranking", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig4_overall_ranking.png", bbox_inches="tight")
    plt.close()
    print(f"✅ 图4：{figures_dir / 'fig4_overall_ranking.png'}")


def main():
    parser = argparse.ArgumentParser(description="GEO / MRGEOA 23指标汇总与绘图")
    parser.add_argument("--input", type=str, default=str(OUTPUT_DIR / "scores" / "evaluation_results_full23.csv"))
    parser.add_argument("--target", type=str, default="m2geo", help="目标方法名，默认 m2geo。如果你的结果里叫 MRGEOA，就传 --target MRGEOA")
    args = parser.parse_args()
    input_csv = Path(args.input)
    target = args.target
    tables_dir, figures_dir = ensure_dirs()
    df, method_mean, method_std, available = load_scores(input_csv)
    print(f"Loaded rows={len(df)}, methods={method_mean['method'].nunique()}, metrics={len(available)}")
    print(f"Target method = {target}")
    save_method_mean_table(method_mean, available, tables_dir)
    adv, recommended = save_advantage_tables(df, method_mean, available, target, tables_dir)
    plot_main_metrics_bar(method_mean, available, figures_dir)
    plot_heatmap(method_mean, available, figures_dir)
    plot_target_advantage(adv, figures_dir)
    plot_overall_ranking(method_mean, available, figures_dir)
    print("\n✅ 汇总与绘图完成")
    print(f"   表格目录：{tables_dir}")
    print(f"   图片目录：{figures_dir}")
    if not recommended.empty:
        print("\n建议优先保留的指标：")
        for _, row in recommended.iterrows():
            print(f" - {row['metric']} ({row['metric_label']}): rank={row['rank']}, gap={row['gap_to_second_or_best']}, win_rate={row['target_win_rate']}")
    else:
        print("\n暂未筛出强优势指标，请查看 mrgeoa_advantage_metrics_full23.csv。")


if __name__ == "__main__":
    main()
