# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==================== 保存位置 ====================
OUTPUT_DIR = Path(__file__).resolve().parent / "figures_53_54"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 全局样式 ====================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 1.1,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

# ==================== 左图数据 ====================
metrics = [
    "Overall",
    "Objective",
    "Subjective",
    "Faithfulness",
    "Utility",
]

kendall_w = np.array([
    0.6043956043956044,
    0.6120879120879121,
    0.5230769230769231,
    0.6148351648351649,
    0.5247252747252747,
])

colors = [
    "#3478B8",  # Overall：蓝色
    "#269C98",  # Objective：青色
    "#6B9C45",  # Subjective：绿色
    "#CE6178",  # Faithfulness：玫红色
    "#A87CC2",  # Utility：紫色
]

# ==================== 右图数据 ====================
# 以下来自原表，尚未通过逐查询记录核验。
baselines = [
    "Terminology",
    "Multi-Agent",
    "Authority",
    "Intent-GEO",
    "Citation",
    "RAG-based",
]

mean_difference = np.array([
    2.03, 2.12, 2.52, 3.97, 3.78, 8.49
])

ci_lower = np.array([
    0.16, 0.65, 0.63, 2.07, 2.17, 6.60
])

ci_upper = np.array([
    4.01, 3.65, 4.44, 5.97, 5.45, 10.42
])

holm_p = [
    "0.444",
    "0.170",
    "0.191",
    "0.004",
    "0.001",
    "<0.001",
]

# ==================== 创建画布 ====================
fig, (ax_a, ax_b) = plt.subplots(
    1, 2,
    figsize=(15, 6.2),
    gridspec_kw={"width_ratios": [1.1, 1.35]},
)

fig.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.96,
    bottom=0.29,
    wspace=0.36,
)

# ==================== 绘制左图 ====================
y_a = np.arange(len(metrics))

ax_a.hlines(
    y=y_a,
    xmin=0,
    xmax=kendall_w,
    colors=colors,
    linewidth=3,
    zorder=2,
)

ax_a.scatter(
    kendall_w,
    y_a,
    c=colors,
    s=75,
    edgecolors="white",
    linewidths=0.6,
    zorder=3,
)

for y, value in zip(y_a, kendall_w):
    ax_a.text(
        value + 0.025,
        y,
        f"{value:.3f}",
        ha="left",
        va="center",
        fontsize=11,
    )

ax_a.set_yticks(y_a)
ax_a.set_yticklabels(metrics)

# 上方留出图例专用空间，避免遮挡数据
ax_a.set_ylim(len(metrics) - 0.5, -2.0)

ax_a.set_xlim(0, 1)
ax_a.set_xticks(np.arange(0, 1.01, 0.2))
ax_a.set_xlabel("Kendall's W", labelpad=9)

# 网格线仅覆盖数据所在区域
for x in np.arange(0.2, 1.0, 0.2):
    ax_a.vlines(
        x,
        ymin=-0.5,
        ymax=len(metrics) - 0.5,
        color="#E5E5E5",
        linewidth=0.7,
        zorder=0,
    )

# ==================== 左图颜色图例 ====================
legend_handles = [
    Line2D(
        [0], [0],
        color=color,
        marker="o",
        linewidth=2.5,
        markersize=6,
        label=metric,
    )
    for metric, color in zip(metrics, colors)
]

legend_a = ax_a.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.025, 0.98),
    ncol=2,
    fontsize=9.5,
    frameon=True,
    fancybox=True,
    framealpha=1.0,
    facecolor="#F1F3F5",
    edgecolor="#333333",
    borderpad=0.8,
    labelspacing=0.6,
    handlelength=2.0,
    handletextpad=0.7,
    columnspacing=1.5,
    borderaxespad=0.0,
)

legend_a.get_frame().set_linewidth(1.0)
legend_a.set_zorder(10)

# ==================== 绘制右图 ====================
y_b = np.arange(len(baselines))

asymmetric_errors = np.vstack([
    mean_difference - ci_lower,
    ci_upper - mean_difference,
])

ax_b.errorbar(
    mean_difference,
    y_b,
    xerr=asymmetric_errors,
    fmt="o",
    color="#3478B8",
    ecolor="#3478B8",
    markersize=7,
    elinewidth=1.8,
    capsize=4,
    capthick=1.5,
    zorder=3,
)

# 零差异参考线，仅覆盖数据区域
ax_b.vlines(
    0,
    ymin=-0.45,
    ymax=len(baselines) - 0.5,
    color="#888888",
    linestyle="--",
    linewidth=1.1,
    zorder=2,
)

ax_b.set_yticks(y_b)
ax_b.set_yticklabels(baselines)

# 上方留出独立图例空间
ax_b.set_ylim(len(baselines) - 0.5, -2.0)

ax_b.set_xlim(-0.5, 13)
ax_b.set_xticks(np.arange(0, 11, 2))

ax_b.set_xlabel(
    "Mean difference (MRGEO − baseline), 95% CI",
    labelpad=9,
)

for x in np.arange(2, 11, 2):
    ax_b.vlines(
        x,
        ymin=-0.45,
        ymax=len(baselines) - 0.5,
        color="#E5E5E5",
        linewidth=0.7,
        zorder=0,
    )

# ==================== p 值列 ====================
p_column_x = 11.65

ax_b.text(
    p_column_x,
    -0.65,
    "Holm p",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)

for y, p_value in zip(y_b, holm_p):
    ax_b.text(
        p_column_x,
        y,
        p_value,
        ha="center",
        va="center",
        fontsize=10,
    )

# ==================== 右图说明图例 ====================
right_handles = [
    Line2D(
        [0], [0],
        color="#3478B8",
        marker="o",
        linestyle="None",
        markersize=6,
        label="Mean difference",
    ),
    Line2D(
        [0], [0],
        color="#3478B8",
        linewidth=1.8,
        marker="|",
        markersize=9,
        label="95% bootstrap CI",
    ),
    Line2D(
        [0], [0],
        color="#888888",
        linestyle="--",
        linewidth=1.1,
        label="Zero difference",
    ),
]

legend_b = ax_b.legend(
    handles=right_handles,
    loc="upper left",
    bbox_to_anchor=(0.025, 0.98),
    ncol=2,
    fontsize=9.5,
    frameon=True,
    fancybox=True,
    framealpha=1.0,
    facecolor="#F1F3F5",
    edgecolor="#333333",
    borderpad=0.8,
    labelspacing=0.6,
    handlelength=2.0,
    handletextpad=0.7,
    columnspacing=1.5,
    borderaxespad=0.0,
)

legend_b.get_frame().set_linewidth(1.0)
legend_b.set_zorder(10)

# ==================== 黑色外框 ====================
for ax in (ax_a, ax_b):
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)

# ==================== 子图标题置于下方 ====================
position_a = ax_a.get_position()
position_b = ax_b.get_position()

fig.text(
    (position_a.x0 + position_a.x1) / 2,
    0.135,
    "(a) Ranking agreement",
    ha="center",
    va="center",
    fontsize=12,
)

fig.text(
    (position_b.x0 + position_b.x1) / 2,
    0.135,
    "(b) Paired performance differences",
    ha="center",
    va="center",
    fontsize=12,
)

# ==================== 保留待核验提示 ====================
fig.text(
    0.5,
    0.045,
    "REVIEW DRAFT: Panel (b) uses supplied statistics "
    "that require query-level verification.",
    ha="center",
    va="center",
    fontsize=9,
    color="#9A4929",
)

# ==================== 保存 ====================
file_stem = "Fig10_Agreement_and_CI_REVIEW_DRAFT"

png_path = OUTPUT_DIR / f"{file_stem}.png"
pdf_path = OUTPUT_DIR / f"{file_stem}.pdf"

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.15,
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.15,
)

plt.close(fig)

print("PNG saved:", png_path)
print("PDF saved:", pdf_path)