"""
data_loader.py
==============
加载样本数据。

默认读取 data/samples.json。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_samples(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    加载样本数据，默认读取 data/samples.json。

    参数：
        path: 可选，指定文件路径。不填则自动找 data/samples.json。

    返回：
        List[Dict]，每个元素包含 sample_id、query、documents 字段。
    """
    base = Path(__file__).resolve().parent

    if path is not None:
        file_path = Path(path)
    else:
        file_path = base / "data" / "samples.json"

    if not file_path.exists():
        raise FileNotFoundError(f"样本文件不存在: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "samples" in data:
        return data["samples"]
    else:
        raise ValueError("samples.json 格式错误，应为 list 或包含 samples 的 dict。")


def load_samples_from_geo_text(txt_path: Path) -> List[Dict[str, Any]]:
    """
    从 GEO_bench_url_cleaned.txt 读取样本并转换为结构化数据。

    文件格式：
    ============================================================
    样本编号: 1
    文档编号: 1
    ============================================================
    URL:
    https://xxx

    CLEANED_TEXT:
    文档内容...

    参数：
        txt_path: GEO_bench_url_cleaned.txt 文件路径

    返回：
        List[Dict]，每个元素包含 sample_id、query、documents 字段
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"源文件不存在: {txt_path}")

    # 读取整个文件
    with txt_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # 按分隔符拆分
    blocks = re.split(r"={60,}", content)

    # 用于存储样本
    samples_dict: Dict[str, List[Dict[str, str]]] = {}  # key: sample_id, value: list of documents

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 提取样本编号
        sample_match = re.search(r"样本编号:\s*(\d+)", block)
        if not sample_match:
            continue
        sample_id = sample_match.group(1)

        # 提取文档编号
        doc_match = re.search(r"文档编号:\s*(\d+)", block)
        doc_id = doc_match.group(1) if doc_match else "1"

        # 提取 URL
        url_match = re.search(r"URL:\s*\n(https?://[^\n]+)", block)
        url = url_match.group(1).strip() if url_match else ""

        # 提取 CLEANED_TEXT
        text_match = re.search(r"CLEANED_TEXT:\s*\n(.*)", block, re.DOTALL)
        text = text_match.group(1).strip() if text_match else ""

        # 构建文档对象
        doc = {
            "doc_id": doc_id,
            "url": url,
            "text": text,
        }

        # 添加到样本字典
        if sample_id not in samples_dict:
            samples_dict[sample_id] = []
        samples_dict[sample_id].append(doc)

    # 转换为最终格式
    samples: List[Dict[str, Any]] = []
    for sample_id, documents in sorted(samples_dict.items(), key=lambda x: int(x[0])):
        first_doc = documents[0] if documents else {"text": ""}

        # 默认 query：通用且不误导具体主题
        query = "Please summarize the key information in these documents."

        # 如果首文档第一行可用，则生成更贴合内容的 query
        first_text = first_doc.get("text", "").strip()
        if first_text:
            first_line = first_text.split("\n", 1)[0].strip()
            if first_line:
                query = f"Tell me about {first_line}"

        sample = {
            "sample_id": sample_id,
            "query": query,
            "documents": documents,
        }
        samples.append(sample)

    return samples


if __name__ == "__main__":
    samples = load_samples()
    print(f"共加载 {len(samples)} 条样本")
    if samples:
        print("第一条预览：")
        first = samples[0]
        print(f"  sample_id: {first.get('sample_id')}")
        print(f"  query: {first.get('query')}")
        print(f"  documents数量: {len(first.get('documents', []))}")
