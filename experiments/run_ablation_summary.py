"""
run_ablation_summary.py
=======================
汇总 MRGEOA / M2GEO 消融实验的 23 指标评分结果。

输入：
    222/outputs/scores/ablation_full23_scores.csv

输出：
    222/outputs/tables/ablation_summary.csv
    222/outputs/tables/ablation_method_rank_overall.csv

运行：
    cd "$HOME\Desktop\111"
    python .\run_ablation_summary.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any


DEFAULT_INPUT = OUTPUT_DIR / "scores" / "ablation_full23_scores.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "tables" / "ablation_summary.csv"
DEFAULT_RANK_OUTPUT = OUTPUT_DIR / "tables" / "ablation_method_rank_overall.csv"

METHOD_ORDER = [
    "MRGEOA-Raw",
    "MRGEOA-S1",
    "MRGEOA-S1S2",
    "MRGEOA-S1S2S3",
    "MRGEOA-S1S2S3S4",
    "MRGEOA-Full",
]

METRICS = [
    "citation_prominence", "answer_dominance", "visibility_overall",
    "objective_score",
    "relevance", "influence", "uniqueness", "diversity", "click_likelihood",
    "subjective_position", "subjective_volume", "subjective_score",
    "attribution_accuracy", "faithfulness", "evidence_precision", "evidence_recall",
    "key_point_coverage", "semantic_contribution", "key_point_recall", "key_point_contradiction",
    "faithfulness_score",
    "clarity", "insight", "coherence", "structure_quality", "readability", "utility_score",
    "overall_score",
]


def safe_float(v: Any):
    try:
        return float(v)
    except Exception:
        return None


def mean(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"评分文件不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method:
            grouped[method].append(row)

    ordered_methods = [m for m in METHOD_ORDER if m in grouped]
    ordered_methods += sorted(m for m in grouped.keys() if m not in METHOD_ORDER)

    summary_rows: List[Dict[str, Any]] = []
    for method in ordered_methods:
        items = grouped[method]
        out: Dict[str, Any] = {"method": method, "n": len(items)}
        for metric in METRICS:
            vals = []
            for item in items:
                v = safe_float(item.get(metric, ""))
                if v is not None:
                    vals.append(v)
            out[metric] = mean(vals)
        summary_rows.append(out)
    return summary_rows


def build_overall_rank(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(summary_rows, key=lambda r: float(r.get("overall_score", 0.0)), reverse=True)
    out = []
    for idx, row in enumerate(ranked, 1):
        out.append({
            "rank": idx,
            "method": row.get("method"),
            "n": row.get("n"),
            "overall_score": row.get("overall_score"),
            "objective_score": row.get("objective_score"),
            "subjective_score": row.get("subjective_score"),
            "faithfulness_score": row.get("faithfulness_score"),
            "utility_score": row.get("utility_score"),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 MRGEOA / M2GEO 消融实验评分结果")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rank-output", type=str, default=str(DEFAULT_RANK_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rank_path = Path(args.rank_output)

    rows = read_csv(input_path)
    summary_rows = summarize(rows)

    fieldnames = ["method", "n"] + METRICS
    write_csv(output_path, summary_rows, fieldnames)

    rank_rows = build_overall_rank(summary_rows)
    write_csv(rank_path, rank_rows, [
        "rank", "method", "n", "overall_score", "objective_score",
        "subjective_score", "faithfulness_score", "utility_score",
    ])

    print(f"✅ 消融汇总完成：{output_path}")
    print(f"✅ overall_score 排名：{rank_path}")


if __name__ == "__main__":
    main()
