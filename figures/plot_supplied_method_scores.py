from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 数据顺序：
# Overall、Objective、Subjective、Faithfulness、Utility
# 以下为统一重算后、保留两位小数的表5数据
data = {
    "MRGEO (Ours)":  [76.22, 73.89, 71.75, 81.14, 81.63],
    "Terminology":   [70.33, 66.88, 66.41, 74.99, 77.26],
    "Multi-Agent":   [69.95, 66.89, 66.74, 73.35, 76.79],
    "Authority":     [69.60, 65.81, 65.70, 72.92, 79.42],
    "Citation":      [68.55, 66.25, 64.01, 73.63, 73.78],
    "Intent-GEO":    [68.36, 64.74, 65.05, 71.47, 77.06],
    "Multimodal":    [67.50, 63.30, 64.66, 70.48, 76.62],
    "Fluency":       [66.41, 62.49, 62.69, 70.16, 75.43],
    "Reputation":    [66.16, 63.65, 63.05, 68.73, 73.11],
    "Uniqueness":    [65.12, 60.78, 66.02, 63.15, 75.29],
    "Statistics":    [64.87, 62.36, 61.95, 66.35, 73.23],
    "RAG-based":     [63.47, 60.68, 58.19, 69.47, 69.60],
    "AutoGEO":       [58.72, 54.79, 54.05, 62.61, 69.41],
    "Simplification": [57.55, 52.92, 55.38, 58.83, 69.03],
}

# 同一种方法在所有图中使用相同颜色
colors = {
    "MRGEO (Ours)": "#2F5D8A",
    "Multi-Agent": "#72B7B2",
    "Intent-GEO": "#54A24B",
    "Terminology": "#B279A2",
    "Multimodal": "#9D755D",
    "Citation": "#E2C14E",
    "Authority": "#E45756",
    "Uniqueness": "#A0A0A0",
    "RAG-based": "#B6992D",
    "Reputation": "#499894",
    "Fluency": "#D37295",
    "Statistics": "#8C8C8C",
    "AutoGEO": "#86B1D1",
    "Simplification": "#C7C7C7",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

output_dir = Path(__file__).resolve().parent / "figures_unified"
output_dir.mkdir(parents=True, exist_ok=True)


def draw_chart(ax, metric, column):
    """按照当前指标降序绘制柱状图。"""
    methods = sorted(
        data,
        key=lambda method: data[method][column],
        reverse=True,
    )
    scores = [data[method][column] for method in methods]

    bars = ax.bar(
        range(len(methods)),
        scores,
        width=0.55,
        color=[colors[method] for method in methods],
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )

    # 柱顶数值
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 1.1,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 横轴名称自然倾斜
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        methods,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", pad=4)

    ax.set_xlabel("Methods", labelpad=10)
    ax.set_ylabel(f"{metric} Score (0-100)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.6,
        alpha=0.8,
    )
    ax.margins(x=0.025)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def save_chart(fig, filename):
    """同时保存高清 PNG 和矢量 PDF。"""
    fig.savefig(output_dir / f"{filename}.png", dpi=600)
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)
    print(f"Saved: {filename}.png / .pdf")


# ==================== 五张单独的图 ====================
plots = [
    ("Overall", 0, "Fig3_Overall"),
    ("Objective", 1, "Fig4a_Objective"),
    ("Faithfulness", 3, "Fig4b_Faithfulness"),
    ("Subjective", 2, "Fig5a_Subjective"),
    ("Utility", 4, "Fig5b_Utility"),
]

for metric, column, filename in plots:
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    draw_chart(ax, metric, column)

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.97,
        bottom=0.29,
    )
    save_chart(fig, filename)


# ==================== 两张并排组合图 ====================
paired_plots = [
    (
        ("Objective", 1),
        ("Faithfulness", 3),
        "Fig4_Objective_Faithfulness",
    ),
    (
        ("Subjective", 2),
        ("Utility", 4),
        "Fig5_Subjective_Utility",
    ),
]

for left, right, filename in paired_plots:
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    for ax, (metric, column), letter in zip(
        axes, [left, right], ["a", "b"]
    ):
        draw_chart(ax, metric, column)
        ax.text(
            0.5,
            -0.49,
            f"({letter}) {metric}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )

    fig.subplots_adjust(
        left=0.045,
        right=0.99,
        top=0.97,
        bottom=0.36,
        wspace=0.18,
    )
    save_chart(fig, filename)

print("\nAll figures saved successfully.")
print("Output folder:", output_dir)