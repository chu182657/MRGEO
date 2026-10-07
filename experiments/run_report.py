"""
run_report.py
=============
汇总评测结果，导出正式图表和表格。
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats

from config import OUTPUT_DIR

# ──────────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────────

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

BASELINE_METHODS = [
    "uniqueness", "simplification", "authority", "fluency",
    "terminology", "reputation", "citation", "statistics"
]
ADVANCED_METHODS = ["autogeo", "intent_geo", "multimodal", "multi_agent", "rag_based"]
M2GEO_METHODS    = ["m2geo"]

OBJECTIVE_METRICS  = ["relevance", "influence", "uniqueness", "diversity", "clickability"]
SUBJECTIVE_METRICS = ["subjective_positivity", "subjective_volubility"]
ALL_METRICS        = OBJECTIVE_METRICS + SUBJECTIVE_METRICS
SUMMARY_METRICS    = ["overall_objective_score", "overall_subjective_score", "overall_score"]

METHOD_LABELS = {
    "uniqueness":     "Uniqueness",
    "simplification": "Simplification",
    "authority":      "Authority",
    "fluency":        "Fluency",
    "terminology":    "Terminology",
    "reputation":     "Reputation",
    "citation":       "Citation",
    "statistics":     "Statistics",
    "autogeo":        "AutoGEO",
    "intent_geo":     "Intent-GEO",
    "multimodal":     "Multimodal",
    "multi_agent":    "Multi-Agent",
    "rag_based":      "RAG-based",
    "m2geo":          "M²GEO",
}

METRIC_LABELS = {
    "relevance":             "Relevance",
    "influence":             "Influence",
    "uniqueness":            "Uniqueness",
    "diversity":             "Diversity",
    "clickability":          "Clickability",
    "subjective_positivity": "Subj. Positivity",
    "subjective_volubility": "Subj. Volubility",
}

GROUP_COLORS = {
    "Baseline": "#4C72B0",
    "Advanced": "#DD8452",
    "M2GEO":    "#55A868",
}

METHOD_ORDER = BASELINE_METHODS + ADVANCED_METHODS + M2GEO_METHODS


def get_method_group(method: str) -> str:
    if method in BASELINE_METHODS:
        return "Baseline"
    elif method in ADVANCED_METHODS:
        return "Advanced"
    return "M2GEO"


def ensure_dirs():
    tables_dir  = OUTPUT_DIR / "tables"
    figures_dir = OUTPUT_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def load_and_prepare(input_csv: Path):
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    for col in ALL_METRICS + SUMMARY_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_metrics = [m for m in ALL_METRICS if m in df.columns]

    method_mean = (
        df.groupby("method")[available_metrics + SUMMARY_METRICS]
        .mean(numeric_only=True)
        .reset_index()
    )
    method_std = (
        df.groupby("method")[available_metrics + SUMMARY_METRICS]
        .std(numeric_only=True)
        .reset_index()
    )

    existing_order = [m for m in METHOD_ORDER if m in method_mean["method"].values]
    other_methods  = [m for m in method_mean["method"].values if m not in METHOD_ORDER]
    final_order    = existing_order + other_methods

    method_mean = method_mean.set_index("method").reindex(final_order).reset_index()
    method_std  = method_std.set_index("method").reindex(final_order).reset_index()

    method_mean["group"] = method_mean["method"].apply(get_method_group)
    method_mean["label"] = method_mean["method"].apply(lambda x: METHOD_LABELS.get(x, x))
    method_std["group"]  = method_std["method"].apply(get_method_group)
    method_std["label"]  = method_std["method"].apply(lambda x: METHOD_LABELS.get(x, x))

    return df, method_mean, method_std, available_metrics


# ──────────────────────────────────────────────
# 表格：主结果表格
# ──────────────────────────────────────────────

def save_main_results_table(method_mean, available_metrics, tables_dir):
    cols = [m for m in available_metrics if m in method_mean.columns]
    if "overall_score" in method_mean.columns:
        cols += ["overall_score"]

    display_df = method_mean[["method", "group"] + cols].copy()
    display_df["method_label"] = display_df["method"].apply(lambda x: METHOD_LABELS.get(x, x))

    csv_path = tables_dir / "main_results_table.csv"
    display_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 主结果表格（CSV）已保存：{csv_path}")

    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Main Results: Comparison of GEO Methods}")
    latex_lines.append("\\label{tab:main_results}")
    col_format = "l|l|" + "c" * len(cols)
    latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_lines.append("\\hline")
    metric_header = " & ".join(
        ["Method", "Group"] + [METRIC_LABELS.get(c, c) for c in cols]
    )
    latex_lines.append(metric_header + " \\\\")
    latex_lines.append("\\hline")

    best_vals   = {}
    second_vals = {}
    for col in cols:
        sorted_vals = display_df[col].dropna().sort_values(ascending=False)
        if len(sorted_vals) >= 1:
            best_vals[col] = sorted_vals.iloc[0]
        if len(sorted_vals) >= 2:
            second_vals[col] = sorted_vals.iloc[1]

    current_group = None
    for _, row in display_df.iterrows():
        group = row["group"]
        if group != current_group:
            if current_group is not None:
                latex_lines.append("\\hline")
            current_group = group
        cells = [row["method_label"], group]
        for col in cols:
            val = row[col]
            if pd.isna(val):
                cells.append("-")
            else:
                formatted = f"{val:.3f}"
                if col in best_vals and abs(val - best_vals[col]) < 1e-6:
                    formatted = f"\\textbf{{{formatted}}}"
                elif col in second_vals and abs(val - second_vals[col]) < 1e-6:
                    formatted = f"\\underline{{{formatted}}}"
                cells.append(formatted)
        latex_lines.append(" & ".join(cells) + " \\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")

    latex_path = tables_dir / "main_results_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    print(f"✅ 主结果表格（LaTeX）已保存：{latex_path}")


# ──────────────────────────────────────────────
# 图1：带误差棒的分组柱状图
# ──────────────────────────────────────────────

def plot_grouped_bar_with_error(method_mean, method_std, available_metrics, figures_dir):
    cols = [m for m in available_metrics if m in method_mean.columns]
    if not cols:
        print("⚠️ 图1跳过：无可用指标列")
        return

    methods   = method_mean["method"].tolist()
    n_methods = len(methods)
    n_metrics = len(cols)
    x         = np.arange(n_metrics)
    width     = 0.6 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(14, n_metrics * 2), 7))

    for i, method in enumerate(methods):
        row_mean = method_mean[method_mean["method"] == method].iloc[0]
        row_std  = method_std[method_std["method"] == method].iloc[0]
        group    = row_mean["group"]
        color    = GROUP_COLORS[group]
        label    = METHOD_LABELS.get(method, method)
        means    = [row_mean[c] if not pd.isna(row_mean[c]) else 0 for c in cols]
        stds     = [row_std[c]  if not pd.isna(row_std[c])  else 0 for c in cols]
        offset   = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, means, width,
               color=color, alpha=0.85,
               yerr=stds, capsize=2,
               error_kw={"elinewidth": 0.8, "alpha": 0.6},
               label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(c, c) for c in cols], rotation=20, ha="right")
    ax.set_ylim(0, 6.0)
    ax.set_ylabel("Mean Score ± Std (1–5)", fontsize=12)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_title("Figure 1: Method Comparison across Metrics (with Error Bars)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, ncol=1)
    sns.despine()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig1_grouped_bar_error.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 图1 已生成：fig1_grouped_bar_error.png")


# ──────────────────────────────────────────────
# 图2：方法-指标热力图
# ──────────────────────────────────────────────

def plot_heatmap(method_mean, available_metrics, figures_dir):
    cols = [m for m in available_metrics if m in method_mean.columns]
    if not cols:
        print("⚠️ 图2跳过：无可用指标列")
        return

    heatmap_df = method_mean.set_index("label")[cols].copy()
    heatmap_df.columns = [METRIC_LABELS.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.5), max(6, len(heatmap_df) * 0.65)))
    sns.heatmap(
        heatmap_df, annot=True, fmt=".2f",
        cmap="RdYlGn", linewidths=0.4,
        ax=ax, vmin=1, vmax=5,
        annot_kws={"size": 9},
        cbar_kws={"label": "Score (1–5)"}
    )
    ax.set_title("Figure 2: Method–Metric Score Heatmap",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig2_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 图2 已生成：fig2_heatmap.png")


# ──────────────────────────────────────────────
# 图3：分组对比图
# ──────────────────────────────────────────────

def plot_group_comparison(method_mean, available_metrics, figures_dir):
    cols = [m for m in available_metrics if m in method_mean.columns]
    if not cols:
        print("⚠️ 图3跳过：无可用指标列")
        return

    group_df = method_mean.groupby("group")[cols].mean().reset_index()
    groups   = ["Baseline", "Advanced", "M2GEO"]
    group_df = group_df.set_index("group").reindex(
        [g for g in groups if g in group_df["group"].values]
    ).reset_index()

    x          = np.arange(len(cols))
    width      = 0.25
    col_labels = [METRIC_LABELS.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (_, row) in enumerate(group_df.iterrows()):
        gname  = row["group"]
        values = [row[c] for c in cols]
        ax.bar(x + i * width, values, width,
               label=gname, color=GROUP_COLORS[gname],
               edgecolor="white", linewidth=0.5)

    ax.set_title("Figure 3: Group-Level Comparison (Baseline vs Advanced vs M²GEO)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Mean Score (1–5)", fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_ylim(0, 5.8)
    ax.legend(title="Group", fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_group_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 图3 已生成：fig3_group_comparison.png")


# ──────────────────────────────────────────────
# 图4：雷达图（已加保护）
# ──────────────────────────────────────────────

def plot_radar(method_mean, available_metrics, figures_dir):
    cols = [m for m in available_metrics if m in method_mean.columns]
    if len(cols) < 3:
        print("⚠️ 图4跳过：指标列不足3个")
        return

    rep_methods = []
    for group, methods in [("Baseline", BASELINE_METHODS), ("Advanced", ADVANCED_METHODS)]:
        sub = method_mean[method_mean["group"] == group].copy()
        if sub.empty:
            continue
        # ✅ 保护：overall_score 全为空时用第一个指标代替
        if "overall_score" in sub.columns and not sub["overall_score"].isna().all():
            best = sub.loc[sub["overall_score"].idxmax(), "method"]
        elif cols and not sub[cols[0]].isna().all():
            best = sub.loc[sub[cols[0]].idxmax(), "method"]
        else:
            best = sub.iloc[0]["method"]
        rep_methods.append(best)

    if "m2geo" in method_mean["method"].values:
        rep_methods.append("m2geo")

    N      = len(cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    col_labels = [METRIC_LABELS.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, method in enumerate(rep_methods):
        row = method_mean[method_mean["method"] == method]
        if row.empty:
            continue
        values = [float(row.iloc[0][c]) if not pd.isna(row.iloc[0][c]) else 1.0 for c in cols]
        values += values[:1]
        label  = METHOD_LABELS.get(method, method)
        color  = colors[i % len(colors)]
        lw     = 3.0 if method == "m2geo" else 1.8
        ax.plot(angles, values, "o-", linewidth=lw, label=label, color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7, color="grey")
    ax.set_title("Figure 4: Radar Chart – Method Capability Profile",
                 fontsize=13, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig4_radar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 图4 已生成：fig4_radar.png")


# ──────────────────────────────────────────────
# 图5：箱线图
# ──────────────────────────────────────────────

def plot_boxplot(df, figures_dir):
    if "overall_score" not in df.columns:
        print("⚠️ 图5跳过：缺少 overall_score 列")
        return

    df_plot = df.dropna(subset=["overall_score"]).copy()
    if df_plot.empty:
        print("⚠️ 图5跳过：overall_score 全为空")
        return

    df_plot["group"] = df_plot["method"].apply(get_method_group)
    df_plot["label"] = df_plot["method"].apply(lambda x: METHOD_LABELS.get(x, x))

    existing    = [m for m in METHOD_ORDER if m in df_plot["method"].unique()]
    label_order = [METHOD_LABELS.get(m, m) for m in existing]

    fig, ax = plt.subplots(figsize=(max(14, len(existing) * 1.1), 6))
    palette = {METHOD_LABELS.get(m, m): GROUP_COLORS[get_method_group(m)] for m in existing}

    sns.boxplot(
        data=df_plot, x="label", y="overall_score",
        order=label_order, palette=palette,
        width=0.6, linewidth=1.2, fliersize=3, ax=ax
    )
    ax.set_title("Figure 5: Distribution of Overall Scores by Method (Box Plot)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Method", fontsize=11)
    ax.set_ylabel("Overall Score (1–5)", fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.tick_params(axis="x", rotation=35)
    patches = [mpatches.Patch(color=v, label=k) for k, v in GROUP_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig5_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 图5 已生成：fig5_boxplot.png")


# ──────────────────────────────────────────────
# 图6：指标相关性热力图
# ──────────────────────────────────────────────

def plot_correlation_heatmap(df, available_metrics, figures_dir):
    cols = [m for m in available_metrics if m in df.columns]
    if len(cols) < 2:
        print("⚠️ 图6跳过：指标列不足2个")
        return

    corr_df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(corr_df) < 3:
        print("⚠️ 图6跳过：有效数据不足")
        return

    corr = corr_df.corr(method="pearson")
    corr.index   = [METRIC_LABELS.get(c, c) for c in corr.index]
    corr.columns = [METRIC_LABELS.get(c, c) for c in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, annot_kws={"size": 10}, ax=ax,
        cbar_kws={"label": "Pearson r"}
    )
    ax.set_title("Figure 6: Metric Correlation Heatmap (Pearson)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig6_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 图6 已生成：fig6_correlation_heatmap.png")


# ──────────────────────────────────────────────
# 图7：显著性检验图
# ──────────────────────────────────────────────

def plot_significance_test(df, figures_dir):
    if "overall_score" not in df.columns or "m2geo" not in df["method"].values:
        print("⚠️ 图7跳过：缺少 overall_score 列或无 m2geo 数据")
        return

    m2geo_scores = df[df["method"] == "m2geo"]["overall_score"].dropna().values
    if len(m2geo_scores) < 3:
        print("⚠️ 图7跳过：m2geo 数据量不足")
        return

    other_methods = [m for m in METHOD_ORDER if m != "m2geo" and m in df["method"].values]
    results = []

    for method in other_methods:
        scores = df[df["method"] == method]["overall_score"].dropna().values
        if len(scores) < 3:
            continue
        try:
            _, p_val = stats.wilcoxon(
                m2geo_scores[:len(scores)],
                scores[:len(m2geo_scores)],
                alternative="greater"
            )
        except Exception:
            p_val = 1.0

        neg_log_p = -np.log10(max(p_val, 1e-10))
        sig = ""
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"

        results.append({
            "method":    METHOD_LABELS.get(method, method),
            "group":     get_method_group(method),
            "neg_log_p": neg_log_p,
            "p_val":     p_val,
            "sig":       sig,
        })

    if not results:
        print("⚠️ 图7跳过：无有效比较结果")
        return

    res_df = pd.DataFrame(results)
    colors = [GROUP_COLORS[g] for g in res_df["group"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(res_df["method"], res_df["neg_log_p"],
                   color=colors, edgecolor="white", linewidth=0.5)

    ax.axvline(-np.log10(0.05),  color="orange",  linestyle="--", linewidth=1.2, label="p=0.05")
    ax.axvline(-np.log10(0.01),  color="red",     linestyle="--", linewidth=1.2, label="p=0.01")
    ax.axvline(-np.log10(0.001), color="darkred", linestyle="--", linewidth=1.2, label="p=0.001")

    for bar, sig in zip(bars, res_df["sig"]):
        if sig:
            ax.text(bar.get_width() + 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    sig, va="center", fontsize=12, color="darkred")

    ax.set_title("Figure 7: Significance Test – M²GEO vs Other Methods\n(Wilcoxon Signed-Rank, one-tailed)",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("-log₁₀(p-value)", fontsize=11)
    ax.set_ylabel("Compared Method", fontsize=11)
    line_handles, _ = ax.get_legend_handles_labels()
    group_patches = [mpatches.Patch(color=v, label=k) for k, v in GROUP_COLORS.items()]
    ax.legend(handles=line_handles + group_patches, loc="lower right", fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig7_significance_test.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 图7 已生成：fig7_significance_test.png")


# ──────────────────────────────────────────────
# 图8：综合得分排名图
# ──────────────────────────────────────────────

def plot_overall_ranking(method_mean, figures_dir):
    if "overall_score" not in method_mean.columns:
        print("⚠️ 图8跳过：缺少 overall_score 列")
        return

    df_sorted = method_mean.dropna(subset=["overall_score"]).sort_values(
        "overall_score", ascending=True
    ).copy()
    if df_sorted.empty:
        print("⚠️ 图8跳过：overall_score 全为空")
        return

    colors = [GROUP_COLORS[g] for g in df_sorted["group"]]
    fig, ax = plt.subplots(figsize=(10, max(5, len(df_sorted) * 0.55)))
    bars = ax.barh(df_sorted["label"], df_sorted["overall_score"],
                   color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, df_sorted["overall_score"]):
        ax.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlim(0, 6.0)
    ax.set_title("Figure 8: Overall Score Ranking of All Methods",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Overall Score (1–5)", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    patches = [mpatches.Patch(color=v, label=k) for k, v in GROUP_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig8_overall_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 图8 已生成：fig8_overall_ranking.png")


# ──────────────────────────────────────────────
# 图9：各指标 Top3 方法
# ──────────────────────────────────────────────

def plot_top3_per_metric(method_mean, available_metrics, figures_dir):
    cols = [m for m in available_metrics if m in method_mean.columns]
    if not cols:
        print("⚠️ 图9跳过：无可用指标列")
        return

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, n * 2.2))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, cols):
        top3   = method_mean.nlargest(3, metric)[["label", metric, "group"]].copy()
        colors = [GROUP_COLORS[g] for g in top3["group"]]
        bars   = ax.barh(top3["label"], top3[metric],
                         color=colors, edgecolor="white")
        for bar, val in zip(bars, top3[metric]):
            ax.text(bar.get_width() + 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, 5.8)
        ax.set_title(f"Top 3 – {METRIC_LABELS.get(metric, metric)}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Score")
        sns.despine(ax=ax)

    plt.suptitle("Figure 9: Top 3 Methods per Metric",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig9_top3_per_metric.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 图9 已生成：fig9_top3_per_metric.png")


# ──────────────────────────────────────────────
# 图10：各方法跨指标折线图
# ──────────────────────────────────────────────

def plot_metric_lines(method_mean, available_metrics, figures_dir):
    cols = [m for m in available_metrics if m in method_mean.columns]
    if not cols:
        print("⚠️ 图10跳过：无可用指标列")
        return

    col_labels = [METRIC_LABELS.get(c, c) for c in cols]
    linestyles = {"Baseline": "-",   "Advanced": "--",  "M2GEO": "-."}
    linewidths = {"Baseline": 1.2,   "Advanced": 1.5,   "M2GEO": 3.0}
    markers    = {"Baseline": "o",   "Advanced": "s",   "M2GEO": "*"}

    fig, ax = plt.subplots(figsize=(13, 6))

    for _, row in method_mean.iterrows():
        method = row["method"]
        group  = row["group"]
        label  = METHOD_LABELS.get(method, method)
        color  = GROUP_COLORS[group]
        values = [float(row[c]) if not pd.isna(row[c]) else np.nan for c in cols]

        ax.plot(col_labels, values,
                linestyle=linestyles[group],
                linewidth=linewidths[group],
                marker=markers[group],
                color=color,
                alpha=1.0 if group == "M2GEO" else 0.75,
                label=label,
                markersize=10 if group == "M2GEO" else 5,
                zorder=10 if group == "M2GEO" else 1)

    ax.set_title("Figure 10: Score Profile across Metrics by Method",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Mean Score (1–5)", fontsize=11)
    ax.set_ylim(0, 5.8)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, ncol=1)
    sns.despine()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig10_metric_lines.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 图10 已生成：fig10_metric_lines.png")


# ──────────────────────────────────────────────
# 排名表
# ──────────────────────────────────────────────

def save_ranking_table(method_mean, tables_dir):
    if "overall_score" not in method_mean.columns:
        return
    ranking = method_mean[["method", "group", "overall_score"]].dropna().copy()
    ranking["method_label"] = ranking["method"].apply(lambda x: METHOD_LABELS.get(x, x))
    ranking = ranking.sort_values("overall_score", ascending=False).reset_index(drop=True)
    ranking.index += 1
    ranking.to_csv(tables_dir / "method_ranking.csv", encoding="utf-8-sig")
    print(f"✅ 排名表已保存：method_ranking.csv")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    input_csv = OUTPUT_DIR / "scores" / "evaluation_results.csv"
    if not input_csv.exists():
        raise FileNotFoundError(
            f"未找到评测结果文件: {input_csv}\n"
            f"请先运行 python run_evaluation.py"
        )

    tables_dir, figures_dir = ensure_dirs()

    print(f"\n📂 读取评测结果：{input_csv}")
    df, method_mean, method_std, available_metrics = load_and_prepare(input_csv)
    print(f"   共 {len(df)} 条记录，{len(method_mean)} 种方法，{len(available_metrics)} 个指标")

    print("\n📋 生成表格...\n")
    save_main_results_table(method_mean, available_metrics, tables_dir)
    save_ranking_table(method_mean, tables_dir)

    print("\n🎨 生成图表...\n")
    plot_grouped_bar_with_error(method_mean, method_std, available_metrics, figures_dir)
    plot_heatmap(method_mean, available_metrics, figures_dir)
    plot_group_comparison(method_mean, available_metrics, figures_dir)
    plot_radar(method_mean, available_metrics, figures_dir)
    plot_boxplot(df, figures_dir)
    plot_correlation_heatmap(df, available_metrics, figures_dir)
    plot_significance_test(df, figures_dir)
    plot_overall_ranking(method_mean, figures_dir)
    plot_top3_per_metric(method_mean, available_metrics, figures_dir)
    plot_metric_lines(method_mean, available_metrics, figures_dir)

    print(f"\n🎉 全部完成！")
    print(f"   表格目录：{tables_dir}")
    print(f"   图片目录：{figures_dir}")


if __name__ == "__main__":
    main()