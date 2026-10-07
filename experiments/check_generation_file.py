"""
check_generation_file.py
========================
评测前检查生成结果文件是否真的包含 14 种方法和 generated_answer。

用法：
    python check_generation_file.py --input outputs/generations/generation_results.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

RESPONSE_KEYS = [
    "generated_answer", "generated_response", "generated_text",
    "generation", "response", "answer", "answer_text",
    "content", "text", "output", "result", "method_output",
    "optimized_text", "rewritten_text", "rewrite", "completion",
    "final_answer",
]

def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"文件为空：{path}")

    if text[0] in "[{":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for k in ["results", "data", "records", "items", "generations"]:
                    if isinstance(data.get(k), list):
                        return data[k]
                return [data]
        except json.JSONDecodeError:
            pass

    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line in ("[", "]"):
            continue
        if line.endswith(","):
            line = line[:-1].strip()
        try:
            rows.append(json.loads(line))
        except Exception as e:
            raise ValueError(f"第 {line_no} 行不是合法 JSON: {line[:120]}") from e
    return rows

def get_response(item):
    for k in RESPONSE_KEYS:
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        raise FileNotFoundError(f"找不到文件：{p}")

    rows = load_json_or_jsonl(p)
    if not rows:
        raise ValueError("没有读取到记录")

    method_counter = Counter(str(r.get("method", "unknown")) for r in rows if isinstance(r, dict))
    sample_ids = {str(r.get("sample_id", "")) for r in rows if isinstance(r, dict)}
    empty_resp = sum(1 for r in rows if isinstance(r, dict) and not get_response(r))
    first = rows[0]

    print("=" * 70)
    print(f"文件: {p}")
    print(f"记录数: {len(rows)}")
    print(f"样本数: {len(sample_ids)}")
    print(f"方法数: {len(method_counter)}")
    print(f"方法分布: {method_counter.most_common(30)}")
    print(f"空 response: {empty_resp} / {len(rows)}")
    print(f"第一条 keys: {list(first.keys()) if isinstance(first, dict) else type(first)}")
    if isinstance(first, dict):
        print(f"第一条 sample_id: {first.get('sample_id')}")
        print(f"第一条 method: {first.get('method')}")
        print(f"第一条 response_len: {len(get_response(first))}")
    print("=" * 70)

    if len(method_counter) < 2 or method_counter.get("unknown", 0) == len(rows) or empty_resp > 0:
        raise SystemExit(
            "❌ 检查不通过：这个文件不是可直接评测的完整生成结果。"
            "不要运行 evaluation。请换正确的 generation_results*.jsonl。"
        )

    print("✅ 检查通过：可以运行 run_evaluation_full23.py")

if __name__ == "__main__":
    main()
