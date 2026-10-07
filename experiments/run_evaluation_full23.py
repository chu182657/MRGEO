"""
run_evaluation_full23.py
========================
Gemini / ChatAnywhere 版本 23 指标评测脚本（防废表版）。

特点：
- 不重新生成文本，只读取已有 generation_results.jsonl；
- 支持 JSONL / JSON 数组；
- 预检查 method / response 是否为空；
- 如果前若干条大量 judge_error，自动停止，避免跑一晚上生成废表；
- 默认输出 outputs/scores/evaluation_results_full23_gemini_SAFE.csv。
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter

from config import OUTPUT_DIR

try:
    from config import REQUEST_INTERVAL
except Exception:
    REQUEST_INTERVAL = 8.0

try:
    from data_loader import load_samples
except Exception:
    load_samples = None

from gemini_client import GeminiClient
from judge_full import JUDGE_METRICS, judge_response


VISIBILITY_METRICS = ["citation_prominence", "answer_dominance", "visibility_overall"]

SUBJECTIVE_METRICS = [
    "relevance", "influence", "uniqueness", "diversity",
    "click_likelihood", "subjective_position", "subjective_volume",
]

FAITHFULNESS_METRICS = [
    "attribution_accuracy", "faithfulness", "evidence_precision", "evidence_recall",
    "key_point_coverage", "semantic_contribution", "key_point_recall",
]

UTILITY_METRICS = ["clarity", "insight", "coherence", "structure_quality", "readability"]

SUMMARY_METRICS = [
    "objective_score", "subjective_score", "faithfulness_score",
    "utility_score", "overall_score",
]

FIELDNAMES = ["sample_id", "method", "query"] + JUDGE_METRICS + SUMMARY_METRICS + ["judge_explanation", "judge_error"]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("run_evaluation_full23")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "evaluation_full23_gemini_SAFE.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到生成结果文件：{path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"生成结果文件为空：{path}")

    if text[0] in "[{":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            if isinstance(data, dict):
                for k in ["results", "data", "records", "items", "generations"]:
                    v = data.get(k)
                    if isinstance(v, list):
                        return [x for x in v if isinstance(x, dict)]
                return [data]
        except json.JSONDecodeError:
            pass

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line in ("[", "]"):
                continue
            if line.endswith(","):
                line = line[:-1].strip()
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行不是合法 JSON：{line[:200]}") from e
            if isinstance(item, dict):
                rows.append(item)
            elif isinstance(item, list):
                rows.extend([x for x in item if isinstance(x, dict)])

    return rows


def get_response(item: Dict[str, Any]) -> str:
    for key in [
        "generated_answer", "generated_response", "generated_text",
        "response", "answer", "answer_text", "method_output",
        "optimized_text", "rewritten_text", "output", "result", "text",
    ]:
        val = item.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def format_documents(docs: Any) -> str:
    if not docs:
        return ""
    if isinstance(docs, str):
        return docs.strip()
    if isinstance(docs, list):
        parts = []
        for i, d in enumerate(docs, 1):
            if isinstance(d, dict):
                text = d.get("text") or d.get("content") or ""
            else:
                text = str(d)
            text = str(text).strip()
            if text:
                parts.append(f"[Doc {i}] {text}")
        return "\n\n".join(parts)
    if isinstance(docs, dict):
        return str(docs.get("text") or docs.get("content") or docs).strip()
    return str(docs).strip()


def build_sample_map() -> Dict[str, Dict[str, Any]]:
    if load_samples is None:
        return {}
    try:
        samples = load_samples()
    except Exception:
        return {}
    mp = {}
    for idx, s in enumerate(samples):
        sid = str(s.get("sample_id", idx + 1))
        mp[sid] = s
    return mp


def get_source_text(item: Dict[str, Any], sample_map: Dict[str, Dict[str, Any]]) -> str:
    for key in ["source_text", "original_text", "raw_text", "document_text", "documents"]:
        if key in item and item.get(key):
            return format_documents(item.get(key))

    sid = str(item.get("sample_id", ""))
    sample = sample_map.get(sid, {})
    if sample:
        for key in ["source_text", "original_text", "raw_text", "document_text", "documents"]:
            if key in sample and sample.get(key):
                return format_documents(sample.get(key))
    return ""


def keep_first_n_sample_ids(records: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    selected = []
    seen = set()
    for item in records:
        sid = str(item.get("sample_id", "")).strip()
        if not sid:
            continue
        if sid not in seen:
            seen.add(sid)
            selected.append(sid)
        if len(selected) >= n:
            break
    allowed = set(selected)
    return [r for r in records if str(r.get("sample_id", "")).strip() in allowed]


def load_existing_keys(output_file: Path) -> Set[Tuple[str, str]]:
    done = set()
    if not output_file.exists():
        return done
    with output_file.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("sample_id", ""))
            method = str(row.get("method", ""))
            err = str(row.get("judge_error", ""))
            overall = str(row.get("overall_score", ""))
            if sid and method and overall and not err.strip():
                done.add((sid, method))
    return done


def mean_safe(values: List[Any]) -> float:
    nums = []
    for v in values:
        try:
            nums.append(float(v))
        except Exception:
            pass
    return round(sum(nums) / len(nums), 4) if nums else 0.0


def add_summary_scores(row: Dict[str, Any]) -> None:
    objective_score = mean_safe([row.get(m, 0) for m in VISIBILITY_METRICS])
    subjective_score = mean_safe([row.get(m, 0) for m in SUBJECTIVE_METRICS])
    contradiction = float(row.get("key_point_contradiction", 0) or 0)
    key_point_consistency = max(0.0, min(5.0, 5.0 - contradiction))
    faithfulness_score = mean_safe([row.get(m, 0) for m in FAITHFULNESS_METRICS] + [key_point_consistency])
    utility_score = mean_safe([row.get(m, 0) for m in UTILITY_METRICS])
    overall_score = round(
        0.30 * objective_score
        + 0.30 * subjective_score
        + 0.25 * faithfulness_score
        + 0.15 * utility_score,
        4,
    )
    row["objective_score"] = objective_score
    row["subjective_score"] = subjective_score
    row["faithfulness_score"] = faithfulness_score
    row["utility_score"] = utility_score
    row["overall_score"] = overall_score


def evaluate_one(item: Dict[str, Any], sample_map: Dict[str, Dict[str, Any]], client: Optional[GeminiClient]) -> Dict[str, Any]:
    sample_id = str(item.get("sample_id", ""))
    method = str(item.get("method", "unknown"))
    query = str(item.get("query", ""))
    response = get_response(item)
    source_text = get_source_text(item, sample_map)

    if not query and sample_id in sample_map:
        query = str(sample_map[sample_id].get("query", ""))

    row = {"sample_id": sample_id, "method": method, "query": query}

    for metric in JUDGE_METRICS:
        row[metric] = 0.0
    row["judge_explanation"] = ""
    row["judge_error"] = ""

    result = judge_response(query=query, response=response, source_text=source_text, method=method, client=client)
    for metric in JUDGE_METRICS:
        row[metric] = result.get(metric, 0.0)
    row["judge_explanation"] = result.get("explanation", "")
    row["judge_error"] = result.get("error", "")

    add_summary_scores(row)
    return row


def save_rows(rows: List[Dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Gemini GEO 23指标评测脚本（防废表版）")
    parser.add_argument("--input", type=str, default=str(OUTPUT_DIR / "generations" / "generation_results.jsonl"))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "scores" / "evaluation_results_full23_gemini_SAFE.csv"))
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--max-initial-errors", type=int, default=10, help="前期错误达到该数量则停止，防止跑废表")
    parser.add_argument("--initial-check-n", type=int, default=30, help="前 N 条用于判断是否大面积失败")
    args = parser.parse_args()

    if args.no_judge:
        raise ValueError("当前脚本用于真实打分，不建议使用 --no-judge。")

    logger = setup_logger()
    input_file = Path(args.input)
    output_file = Path(args.output)

    logger.info("=" * 70)
    logger.info("Gemini GEO 23指标评测开始")
    logger.info("读取生成结果：%s", input_file)

    records = load_json_or_jsonl(input_file)
    logger.info("原始记录数：%s", len(records))

    records = keep_first_n_sample_ids(records, args.max_samples)

    empty_response = sum(1 for r in records if not get_response(r))
    method_counter = Counter(str(r.get("method", "unknown")) for r in records)

    logger.info("保留前 %s 个 sample_id 后，记录数：%s", args.max_samples, len(records))
    logger.info("样本数：%s，方法数：%s", len({str(r.get('sample_id','')) for r in records}), len(method_counter))
    logger.info("方法分布：%s", method_counter.most_common(20))
    logger.info("空 response 记录数：%s / %s", empty_response, len(records))

    if not records:
        raise ValueError("没有可评测记录。")
    if empty_response == len(records):
        raise ValueError("全部 response 为空，输入文件不是生成结果文件。")
    if "unknown" in method_counter and len(method_counter) == 1:
        raise ValueError("method 全部为 unknown，输入文件字段不对。")

    sample_map = build_sample_map()
    logger.info("原始样本映射数：%s", len(sample_map))

    done_keys = load_existing_keys(output_file) if args.resume else set()
    if args.resume:
        logger.info("断点续跑：已有完成记录 %s 条", len(done_keys))

    rows = []
    if args.resume and output_file.exists():
        with output_file.open("r", encoding="utf-8-sig") as f:
            rows.extend(list(csv.DictReader(f)))

    client = GeminiClient()

    total = len(records)
    skipped = 0
    failed = 0
    initial_errors = 0
    processed_new = 0

    for idx, item in enumerate(records, start=1):
        sample_id = str(item.get("sample_id", ""))
        method = str(item.get("method", "unknown"))
        key = (sample_id, method)

        if args.resume and key in done_keys:
            skipped += 1
            continue

        row = evaluate_one(item, sample_map, client)
        rows.append(row)
        processed_new += 1

        if str(row.get("judge_error", "")).strip():
            failed += 1
            if processed_new <= args.initial_check_n:
                initial_errors += 1

        save_rows(rows, output_file)

        if processed_new == args.initial_check_n and initial_errors >= args.max_initial_errors:
            raise RuntimeError(
                f"前 {args.initial_check_n} 条里已有 {initial_errors} 条 judge_error，"
                f"已自动停止，避免生成废表。请先检查模型/Prompt/接口。"
            )

        if idx % 20 == 0 or idx == total:
            logger.info("进度：%s/%s，跳过=%s，judge_error=%s，输出=%s", idx, total, skipped, failed, output_file)

        time.sleep(float(REQUEST_INTERVAL))

    logger.info("评测完成：输出 %s 条记录到 %s", len(rows), output_file)
    print(f"\n✅ 23指标评测完成：{output_file}")
    print(f"   记录数：{len(rows)} | 跳过：{skipped} | judge_error：{failed}")


if __name__ == "__main__":
    main()
