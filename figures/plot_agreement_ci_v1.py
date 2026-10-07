# -*- coding: utf-8 -*-
# 运行后生成一张左右并排的图，(a)、(b) 及子图名称均位于子图下方。
# 右图使用原表提供的统计值，尚未通过逐查询数据核验，因此保留草稿提示。

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ==================== 保存位置 ====================
OUTPUT_DIR = Path(__file__).resolve().parent / "figures_53_54"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 样式 ====================
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

# ==================== 左图：Kendall's W ====================
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
    "#3478B8",
    "#269C98",
    "#6B9C45",
    "#CE6178",
    "#A87CC2",
]

# ==================== 右图：原表提供的待核验统计 ====================
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

# ==================== 创建并列画布 ====================
fig, (ax_a, ax_b) = plt.subplots(
    1, 2,
    figsize=(14, 5.6),
    gridspec_kw={"width_ratios": [1, 1.35]},
)

# 为横轴名称、下方子图标题和底部核验提示分别预留空间
fig.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.96,
    bottom=0.30,
    wspace=0.40,
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
ax_a.set_ylim(len(metrics) - 0.5, -0.5)
ax_a.set_xlim(0, 1)
ax_a.set_xticks(np.arange(0, 1.01, 0.2))
ax_a.set_xlabel("Kendall's W", labelpad=9)

ax_a.grid(
    axis="x",
    color="#E5E5E5",
    linewidth=0.7,
)
ax_a.set_axisbelow(True)

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

ax_b.axvline(
    0,
    color="#888888",
    linestyle="--",
    linewidth=1.1,
    zorder=2,
)

ax_b.set_yticks(y_b)
ax_b.set_yticklabels(baselines)
ax_b.set_ylim(len(baselines) - 0.5, -1.0)
ax_b.set_xlim(-0.5, 13)
ax_b.set_xticks(np.arange(0, 11, 2))

ax_b.set_xlabel(
    "Mean difference (MRGEO − baseline), 95% CI",
    labelpad=9,
)

# p 值列放在坐标框内
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

ax_b.grid(
    axis="x",
    color="#E5E5E5",
    linewidth=0.7,
)
ax_b.set_axisbelow(True)

# ==================== 黑色坐标边框 ====================
for ax in (ax_a, ax_b):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)

# ==================== 子图标题：放在下方 ====================
# 使用画布坐标，使两侧标题处于同一水平位置
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

# 待核验说明不占用子图标题位置
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

# ==================== 保存 PNG 和 PDF ====================
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