"""
filter_empty_generation.py
==========================
从 generation_results.jsonl 中删除空 generated_answer/response 记录，
并输出一份空记录清单，避免 run_evaluation_full23.py 因空 response 停止。

用法：
    python filter_empty_generation.py --input outputs/generations/generation_results.jsonl --output outputs/generations/generation_results_nonempty.jsonl
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


RESPONSE_KEYS = [
    "generated_answer",
    "generated_response",
    "generated_text",
    "response",
    "answer",
    "answer_text",
    "content",
    "text",
    "output",
    "result",
    "method_output",
    "optimized_text",
    "rewritten_text",
    "final_answer",
]


def get_response(item: Dict[str, Any]) -> str:
    for key in RESPONSE_KEYS:
        val = item.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] in "[{":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            if isinstance(data, dict):
                for key in ["results", "data", "records", "items", "generations"]:
                    val = data.get(key)
                    if isinstance(val, list):
                        return [x for x in val if isinstance(x, dict)]
                return [data]
        except json.JSONDecodeError:
            pass

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"):
                continue
            if line.endswith(","):
                line = line[:-1].strip()
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except Exception:
                continue
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/generations/generation_results_nonempty.jsonl")
    parser.add_argument("--bad", default="outputs/generations/empty_generation_records.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    bad_path = Path(args.bad)

    records = load_records(input_path)

    good = []
    bad = []

    for item in records:
        resp = get_response(item)
        method = str(item.get("method", "unknown"))
        sample_id = str(item.get("sample_id", ""))
        if resp:
            good.append(item)
        else:
            bad.append({
                "sample_id": sample_id,
                "method": method,
                "query": str(item.get("query", ""))[:200],
                "keys": ",".join(item.keys()),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for item in good:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with bad_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "method", "query", "keys"])
        writer.writeheader()
        writer.writerows(bad)

    total = len(records)
    print("=" * 70)
    print(f"输入文件: {input_path}")
    print(f"总记录数: {total}")
    print(f"有效记录数: {len(good)}")
    print(f"空 response 记录数: {len(bad)}")
    print(f"有效文件已输出: {output_path}")
    print(f"空记录清单已输出: {bad_path}")
    print("-" * 70)
    print("原始方法分布:")
    for k, v in Counter(str(x.get("method", "unknown")) for x in records).most_common():
        print(f"  {k}: {v}")
    print("-" * 70)
    print("空 response 方法分布:")
    for k, v in Counter(str(x.get("method", "unknown")) for x in bad).most_common():
        print(f"  {k}: {v}")

    if len(good) == 0:
        raise RuntimeError("没有任何有效记录，不能继续评测。")

    print("=" * 70)
    print("✅ 过滤完成。下一步请用输出的 nonempty 文件跑 evaluation。")


if __name__ == "__main__":
    main()
