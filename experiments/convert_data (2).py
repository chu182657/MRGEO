"""
data_loader.py
==============
加载并清洗样本数据。

功能：
1) load_samples: 读取 JSON 样本（data/samples.json 或自定义路径）
2) load_samples_from_geo_text: 从 GEO_bench_url_cleaned.txt 解析为结构化样本
3) 统一数据结构，尽量在“进入模型前”做清洗，减少无效 token 消耗
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# 可调参数（按需修改）
# -----------------------------
DEFAULT_JSON_REL_PATH = Path("data") / "samples.json"
MAX_QUERY_LEN = 200
MAX_TITLE_LEN = 120


def _safe_str(x: Any) -> str:
    """把任意对象稳健转成字符串并去首尾空白。"""
    if x is None:
        return ""
    return str(x).strip()


def _normalize_doc(doc: Any, idx: int) -> Optional[Dict[str, str]]:
    """
    归一化单个文档：
    - 保证返回字段 doc_id/url/text
    - 空文档（url/text都空）返回 None
    """
    if not isinstance(doc, dict):
        return None

    doc_id = _safe_str(doc.get("doc_id", idx + 1))
    url = _safe_str(doc.get("url", ""))
    text = _safe_str(doc.get("text", ""))

    if not url and not text:
        return None

    return {"doc_id": doc_id, "url": url, "text": text}


def _build_query_from_documents(sample_id: str, documents: List[Dict[str, str]]) -> str:
    """
    若原始 query 缺失时，为样本构造保守 query：
    - 优先取首篇文档第一行（像标题时）
    - 否则使用通用 query
    """
    default_q = f"Please summarize key information for sample {sample_id}."

    if not documents:
        return default_q

    first_text = _safe_str(documents[0].get("text", ""))
    if not first_text:
        return default_q

    first_line = _safe_str(first_text.split("\n", 1)[0])

    # 基础过滤：过短/过长/疑似噪声
    if 4 <= len(first_line) <= MAX_TITLE_LEN:
        # 去掉过多空白
        first_line = re.sub(r"\s+", " ", first_line)
        return f"Tell me about: {first_line}"

    return default_q


def _normalize_sample(item: Any, fallback_id: int) -> Optional[Dict[str, Any]]:
    """
    归一化单条样本：
    输出字段固定为 sample_id/query/documents。
    """
    if not isinstance(item, dict):
        return None

    sample_id = _safe_str(item.get("sample_id", fallback_id))
    query = _safe_str(item.get("query", ""))

    raw_docs = item.get("documents", [])
    if not isinstance(raw_docs, list):
        raw_docs = []

    documents: List[Dict[str, str]] = []
    for i, d in enumerate(raw_docs):
        nd = _normalize_doc(d, i)
        if nd is not None:
            documents.append(nd)

    if not query:
        query = _build_query_from_documents(sample_id, documents)

    # query 限长，避免脏数据导致 token 暴涨
    if len(query) > MAX_QUERY_LEN:
        query = query[:MAX_QUERY_LEN].rstrip()

    return {
        "sample_id": sample_id,
        "query": query,
        "documents": documents,
    }


def load_samples(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    加载样本数据（JSON）。

    参数：
        path: 可选，指定 json 文件路径。不填则使用 data/samples.json

    返回：
        List[Dict]，每个元素固定包含：
        - sample_id: str
        - query: str
        - documents: List[{"doc_id","url","text"}]
    """
    base = Path(__file__).resolve().parent
    file_path = Path(path) if path else (base / DEFAULT_JSON_REL_PATH)

    if not file_path.exists():
        raise FileNotFoundError(f"样本文件不存在: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_samples = data
    elif isinstance(data, dict) and isinstance(data.get("samples"), list):
        raw_samples = data["samples"]
    else:
        raise ValueError("samples.json 格式错误：应为 list，或包含 list 类型的 samples 字段。")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_samples, start=1):
        ns = _normalize_sample(item, idx)
        if ns is not None:
            normalized.append(ns)

    return normalized


def load_samples_from_geo_text(txt_path: Path) -> List[Dict[str, Any]]:
    """
    从 GEO_bench_url_cleaned.txt 读取样本并转换为结构化数据。

    期望片段示例：
    ============================================================
    样本编号: 1
    文档编号: 1
    ============================================================
    URL:
    https://xxx

    CLEANED_TEXT:
    文档内容...
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"源文件不存在: {txt_path}")

    with txt_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # 更稳：按“出现样本编号”的位置切块，而不是强依赖分隔线长度
    blocks = re.split(r"(?=样本编号:\s*\d+)", content)
    samples_dict: Dict[str, List[Dict[str, str]]] = {}

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        sample_match = re.search(r"样本编号:\s*(\d+)", block)
        if not sample_match:
            continue
        sample_id = sample_match.group(1)

        doc_match = re.search(r"文档编号:\s*(\d+)", block)
        doc_id = doc_match.group(1) if doc_match else "1"

        # URL 部分（允许 URL 后面有空行）
        url_match = re.search(r"URL:\s*\n([^\n]+)", block)
        url = _safe_str(url_match.group(1)) if url_match else ""
        if url and not re.match(r"^https?://", url):
            url = ""

        # CLEANED_TEXT 到块尾
        text_match = re.search(r"CLEANED_TEXT:\s*\n(.*)", block, flags=re.DOTALL)
        text = _safe_str(text_match.group(1)) if text_match else ""

        doc = _normalize_doc({"doc_id": doc_id, "url": url, "text": text}, 0)
        if doc is None:
            continue

        samples_dict.setdefault(sample_id, []).append(doc)

    # 组装样本并做统一归一化
    samples: List[Dict[str, Any]] = []
    for sid in sorted(samples_dict.keys(), key=lambda x: int(x)):
        item = {
            "sample_id": sid,
            "query": "",  # 交给 normalize 自动生成
            "documents": samples_dict[sid],
        }
        ns = _normalize_sample(item, int(sid))
        if ns is not None:
            samples.append(ns)

    return samples


if __name__ == "__main__":
    samples = load_samples()
    print(f"共加载 {len(samples)} 条样本")

    if not samples:
        print("样本为空，请检查数据文件。")
    else:
        first = samples[0]
        print("第一条预览：")
        print(f"  sample_id: {first.get('sample_id')}")
        print(f"  query: {first.get('query')}")
        print(f"  documents数量: {len(first.get('documents', []))}")
