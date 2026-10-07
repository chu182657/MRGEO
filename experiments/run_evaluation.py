"""
run_evaluation.py
=================
读取生成结果，计算所有指标并保存评测结果。

自动指标：
    - word_count            词数
    - positive_word_count   积极词汇数量
    - unique_word_ratio     唯一词占比

LLM 评分（7个维度）：
    客观指标：
    - relevance             相关性
    - influence             影响力
    - uniqueness            独特性
    - diversity             多样性
    - clickability          吸引力
    主观指标：
    - subjective_positivity 主观积极性
    - subjective_volubility 主观表达丰富度

汇总分：
    - overall_objective_score
    - overall_subjective_score
    - overall_score
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import ENABLE_JUDGE, OUTPUT_DIR
from judge import judge_response
from metrics import tokenize, word_count, unique_word_ratio


# ──────────────────────────────────────────────
# 积极词表（用于 positive_word_count）
# ──────────────────────────────────────────────

POSITIVE_WORDS = {
    # 英文积极词
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "outstanding", "best", "top", "leading", "successful", "famous",
    "popular", "important", "significant", "notable", "remarkable",
    "impressive", "valuable", "effective", "powerful", "strong",
    "beautiful", "unique", "innovative", "advanced", "superior",
    "celebrated", "renowned", "prestigious", "award", "champion",
    "victory", "win", "achievement", "success", "proud",
    # 中文积极词
    "优秀", "出色", "卓越", "杰出", "著名", "重要", "显著",
    "成功", "领先", "最佳", "顶尖", "独特", "创新", "先进",
    "强大", "有效", "美好", "辉煌", "荣誉", "冠军", "胜利",
}


def count_positive_words(text: str) -> int:
    """统计文本中积极词汇的数量。"""
    if not text:
        return 0
    tokens = set(tokenize(text))
    return sum(1 for w in POSITIVE_WORDS if w in tokens)


# ──────────────────────────────────────────────
# 日志
# ──────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("run_evaluation")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "evaluation.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# ──────────────────────────────────────────────
# 读取生成结果
# ──────────────────────────────────────────────

def load_generation_records(input_path: Path) -> List[Dict[str, Any]]:
    """读取生成结果文件，支持 jsonl 和 json 格式。"""
    if not input_path.exists():
        raise FileNotFoundError(f"生成结果文件不存在: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".jsonl":
        records = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict) and "results" in loaded:
            return loaded["results"]
        raise ValueError("JSON 格式错误")

    raise ValueError(f"不支持的文件类型: {suffix}")


# ──────────────────────────────────────────────
# 评测单条记录
# ──────────────────────────────────────────────

def evaluate_record(item: Dict[str, Any], enable_judge: bool) -> Dict[str, Any]:
    """对单条生成结果计算所有指标。"""
    sample_id = str(item.get("sample_id", ""))
    method    = str(item.get("method", "unknown"))
    query     = str(item.get("query", ""))
    response  = str(item.get("generated_answer") or item.get("response") or "")

    # ── 自动指标 ──
    wc  = word_count(response)
    pwc = count_positive_words(response)
    uwr = round(unique_word_ratio(response), 6)

    row: Dict[str, Any] = {
        "sample_id":          sample_id,
        "method":             method,
        "word_count":         wc,
        "positive_word_count": pwc,
        "unique_word_ratio":  uwr,
        # LLM 评分占位
        "relevance":                "",
        "influence":                "",
        "uniqueness":               "",
        "diversity":                "",
        "clickability":             "",
        "subjective_positivity":    "",
        "subjective_volubility":    "",
        "overall_objective_score":  "",
        "overall_subjective_score": "",
        "overall_score":            "",
        "judge_explanation":        "",
        "judge_error":              "",
    }

    # ── LLM 评分 ──
    if enable_judge:
        result = judge_response(query=query, response=response)
        row["relevance"]                = result.get("relevance", "")
        row["influence"]                = result.get("influence", "")
        row["uniqueness"]               = result.get("uniqueness", "")
        row["diversity"]                = result.get("diversity", "")
        row["clickability"]             = result.get("clickability", "")
        row["subjective_positivity"]    = result.get("subjective_positivity", "")
        row["subjective_volubility"]    = result.get("subjective_volubility", "")
        row["overall_objective_score"]  = result.get("overall_objective_score", "")
        row["overall_subjective_score"] = result.get("overall_subjective_score", "")
        row["overall_score"]            = result.get("overall_score", "")
        row["judge_explanation"]        = result.get("explanation", "")
        row["judge_error"]              = result.get("error", "")

    return row


# ──────────────────────────────────────────────
# 保存 CSV
# ──────────────────────────────────────────────

FIELDNAMES = [
    "sample_id", "method",
    "word_count", "positive_word_count", "unique_word_ratio",
    "relevance", "influence", "uniqueness", "diversity", "clickability",
    "subjective_positivity", "subjective_volubility",
    "overall_objective_score", "overall_subjective_score", "overall_score",
    "judge_explanation", "judge_error",
]


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    logger = setup_logger()

    # 找生成结果文件
    jsonl_path = OUTPUT_DIR / "generations" / "generation_results.jsonl"
    json_path  = OUTPUT_DIR / "generation_results.json"

    if jsonl_path.exists():
        input_file = jsonl_path
    elif json_path.exists():
        input_file = json_path
    else:
        raise FileNotFoundError(
            f"未找到生成结果文件，请先运行 python run_generation.py\n"
            f"查找路径：{jsonl_path}"
        )

    logger.info("开始评测，输入文件: %s", input_file)
    records = load_generation_records(input_file)
    logger.info("读取到 %s 条生成记录", len(records))

    rows = []
    for idx, item in enumerate(records, start=1):
        try:
            row = evaluate_record(item, enable_judge=ENABLE_JUDGE)
            rows.append(row)
            if idx % 50 == 0:
                logger.info("评测进度: %s/%s", idx, len(records))
        except Exception as e:
            logger.exception(
                "评测失败 sample_id=%s method=%s error=%s",
                item.get("sample_id"), item.get("method"), e
            )
            continue

    output_file = OUTPUT_DIR / "scores" / "evaluation_results.csv"
    save_csv(rows, output_file)
    logger.info("评测完成，共写入 %s 条，输出: %s", len(rows), output_file)
    print(f"\n✅ 评测完成！共 {len(rows)} 条记录")
    print(f"   结果保存在：{output_file}")


if __name__ == "__main__":
    main()