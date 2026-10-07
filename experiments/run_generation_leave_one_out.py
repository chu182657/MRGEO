r"""
run_generation_leave_one_out.py
===============================

MRGEO 去层消融实验生成脚本。

用途：
    生成完整 MRGEO 与去掉单层后的对照结果，用于论文 5.3.2 去层消融实验。

实验组：
    - MRGEO-Full
    - MRGEO-w/o-S1
    - MRGEO-w/o-S2
    - MRGEO-w/o-S3
    - MRGEO-w/o-S4
    - MRGEO-w/o-S5

默认输入：
    当前项目 data/samples.json

默认输出：
    Desktop/222/outputs/generations/leave_one_out_generation_results.jsonl

运行示例：
    python .\run_generation_leave_one_out.py --max-samples 3 --resume
    python .\run_generation_leave_one_out.py --max-samples 50 --resume

后续评分：
    python .\run_evaluation_full23.py ^
      --input "$HOME\Desktop\222\outputs\generations\leave_one_out_generation_results.jsonl" ^
      --output "$HOME\Desktop\222\outputs\scores\leave_one_out_full23_scores.csv" ^
      --max-samples 999999 ^
      --resume
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

from config import OUTPUT_DIR

try:
    from config import REQUEST_INTERVAL
except Exception:
    REQUEST_INTERVAL = 0.5

from data_loader import load_samples
from deepseek_client import DeepSeekClient


# ============================================================
# 路径设置
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_DIR = OUTPUT_DIR
DEFAULT_OUTPUT_FILE = DEFAULT_TARGET_DIR / "generations" / "leave_one_out_generation_results.jsonl"
DEFAULT_LOG_FILE = DEFAULT_TARGET_DIR / "logs" / "generation_leave_one_out.log"


# ============================================================
# 日志
# ============================================================

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("run_generation_leave_one_out")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# 样本字段处理
# ============================================================

def extract_fields(sample: Dict[str, Any]) -> Tuple[str, List[str]]:
    """从 sample 中提取 query 和 documents，兼容 list[str] / list[dict]。"""
    query = str(sample.get("query", "")).strip()

    raw_docs = sample.get("documents", [])
    if not isinstance(raw_docs, list):
        raw_docs = [raw_docs]

    docs: List[str] = []
    for item in raw_docs:
        if isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("content")
                or item.get("cleaned_text")
                or ""
            )
        else:
            text = str(item)
        text = str(text).strip()
        if text:
            docs.append(text)

    return query, docs


def format_docs(docs: List[str]) -> str:
    """将文档列表格式化为 [Doc X] 文本。"""
    if not docs:
        return "（无可用参考资料）"
    return "\n\n".join(f"[Doc {idx}]\n{doc}" for idx, doc in enumerate(docs, start=1))


def get_source_text(sample: Dict[str, Any]) -> str:
    _, docs = extract_fields(sample)
    return format_docs(docs)


# ============================================================
# MRGEO 五层定义
# ============================================================

LAYER_NAMES = {
    "S1": "内容准入 Content Admission",
    "S2": "生成偏好 Generation Preference",
    "S3": "语义一致性 Semantic Consistency",
    "S4": "多特征增强 Multi-feature Enhancement",
    "S5": "意图对齐 Intent Alignment",
}

LAYER_INSTRUCTIONS = {
    "S1": """【第 1 层：内容准入 Content Admission】
从所有参考资料中筛选出与用户问题直接相关的内容片段，剔除无关或弱相关信息。
要求：只保留能够直接支撑回答的事实、观点、数据或解释，不引入资料外信息。""",
    "S2": """【第 2 层：生成偏好 Generation Preference】
根据生成式引擎更容易采用的回答结构组织信息。
要求：优先呈现直接回答问题的核心信息，其次补充支撑性背景、证据、例子或解释，避免把次要信息放在显著位置。""",
    "S3": """【第 3 层：语义一致性 Semantic Consistency】
检查最终内容与参考资料之间的语义一致性。
要求：不歪曲原文含义，不夸大事实，不遗漏关键限制，不制造资料中不存在的因果关系或结论。""",
    "S4": """【第 4 层：多特征增强 Multi-feature Enhancement】
在忠实于资料的前提下增强内容的 GEO 可见性特征。
要求：自然使用资料中的数字、时间、地点、分类、列表、因果关系、对比关系、关键事实和来源标记，使回答更具体、更有结构感和信息密度。""",
    "S5": """【第 5 层：意图对齐 Intent Alignment】
根据用户问题判断其核心检索意图，并让最终文章直接回应这一意图。
要求：开头直接回答核心问题，中间给出依据和必要背景，结尾简洁收束，避免偏离用户真正想问的内容。""",
}

FULL_ORDER = ["S1", "S2", "S3", "S4", "S5"]


# ============================================================
# Prompt 构造：Full + Leave-one-out
# ============================================================

def build_layer_prompt(sample: Dict[str, Any], active_layers: List[str], removed_layer: str = "") -> str:
    """
    根据 active_layers 构造 MRGEO 实验 prompt。
    removed_layer 仅用于在去层实验中明确标记被移除的层，避免模型自行补回该步骤。
    """
    query, docs = extract_fields(sample)
    formatted = format_docs(docs)

    layer_text = "\n\n".join(LAYER_INSTRUCTIONS[layer] for layer in active_layers)
    layer_sequence = " → ".join(f"{layer}（{LAYER_NAMES[layer]}）" for layer in active_layers)

    if removed_layer:
        removed_note = (
            f"\n【去层设置】\n本实验刻意移除 {removed_layer}（{LAYER_NAMES[removed_layer]}）。"
            f"请不要显式执行该层的专门流程，也不要在回答中说明去掉了该层。\n"
        )
        task_name = f"MRGEO 去层消融实验：移除 {removed_layer}"
    else:
        removed_note = ""
        task_name = "完整 MRGEO 五层递进实验"

    return f"""你是一个高级问答优化系统，现在执行{task_name}。

【参考资料】
{formatted}

【用户问题】
{query}

【执行顺序】
{layer_sequence}
{removed_note}
【需要执行的层】
{layer_text}

【最终文章要求】
1. 只输出最终文章，不输出分析过程、步骤标题或自我解释。
2. 全文必须基于参考资料，不得编造资料中不存在的事实。
3. 回答应自然、清楚、完整，长度约 250-450 字。
4. 保持客观、可信、结构清晰，适合生成式引擎回答场景。

请直接输出最终文章。"""


def build_full_prompt(sample: Dict[str, Any]) -> str:
    return build_layer_prompt(sample, FULL_ORDER, removed_layer="")


def build_without_s1_prompt(sample: Dict[str, Any]) -> str:
    return build_layer_prompt(sample, ["S2", "S3", "S4", "S5"], removed_layer="S1")


def build_without_s2_prompt(sample: Dict[str, Any]) -> str:
    return build_layer_prompt(sample, ["S1", "S3", "S4", "S5"], removed_layer="S2")


def build_without_s3_prompt(sample: Dict[str, Any]) -> str:
    return build_layer_prompt(sample, ["S1", "S2", "S4", "S5"], removed_layer="S3")


def build_without_s4_prompt(sample: Dict[str, Any]) -> str:
    return build_layer_prompt(sample, ["S1", "S2", "S3", "S5"], removed_layer="S4")


def build_without_s5_prompt(sample: Dict[str, Any]) -> str:
    return build_layer_prompt(sample, ["S1", "S2", "S3", "S4"], removed_layer="S5")


GENERATION_METHODS: List[Tuple[str, Callable[[Dict[str, Any]], str]]] = [
    ("MRGEO-Full", build_full_prompt),
    ("MRGEO-w/o-S1", build_without_s1_prompt),
    ("MRGEO-w/o-S2", build_without_s2_prompt),
    ("MRGEO-w/o-S3", build_without_s3_prompt),
    ("MRGEO-w/o-S4", build_without_s4_prompt),
    ("MRGEO-w/o-S5", build_without_s5_prompt),
]


# ============================================================
# 断点续跑与写入
# ============================================================

def load_done_keys(output_file: Path) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    if not output_file.exists():
        return done

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            sample_id = str(item.get("sample_id", "")).strip()
            method = str(item.get("method", "")).strip()
            answer = str(item.get("generated_answer") or item.get("response") or "").strip()
            if sample_id and method and answer:
                done.add((sample_id, method))
    return done


def append_jsonl(output_file: Path, row: Dict[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


# ============================================================
# 单条生成
# ============================================================

def generate_one(
    client: DeepSeekClient,
    sample: Dict[str, Any],
    method_name: str,
    prompt: str,
) -> Dict[str, Any]:
    sample_id = str(sample.get("sample_id", ""))
    query = str(sample.get("query", ""))

    result = client.send_request(prompt)
    if result.get("success"):
        answer = str(result.get("text", "")).strip()
        error = ""
    else:
        answer = ""
        error = str(result.get("error", "DeepSeek 调用失败"))

    return {
        "sample_id": sample_id,
        "method": method_name,
        "query": query,
        "generated_answer": answer,
        "response": answer,
        "source_text": get_source_text(sample),
        "judge_error": "",
        "generation_error": error,
    }


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="MRGEO 去层消融实验生成脚本")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE), help="生成结果 JSONL 输出路径")
    parser.add_argument("--max-samples", type=int, default=999999, help="最多运行多少个样本")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已成功生成的 sample_id + method")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件")
    parser.add_argument("--only-method", type=str, default="", help="只运行某一个方法，例如 MRGEO-w/o-S3")
    args = parser.parse_args()

    output_file = Path(args.output)
    logger = setup_logger(DEFAULT_LOG_FILE)

    logger.info("=" * 80)
    logger.info("MRGEO 去层消融实验生成开始")
    logger.info("项目目录：%s", PROJECT_DIR)
    logger.info("输出文件：%s", output_file)

    if output_file.exists() and not args.resume and not args.overwrite:
        raise RuntimeError(
            f"输出文件已存在：{output_file}\n"
            f"如需断点续跑，请加 --resume；如需覆盖，请加 --overwrite。"
        )
    if output_file.exists() and args.overwrite:
        logger.warning("你使用了 --overwrite，已有输出文件将被删除：%s", output_file)
        output_file.unlink()

    samples = load_samples()
    samples = samples[: max(0, int(args.max_samples))]

    selected_methods = GENERATION_METHODS
    if args.only_method.strip():
        selected_methods = [item for item in GENERATION_METHODS if item[0] == args.only_method.strip()]
        if not selected_methods:
            supported = ", ".join([m for m, _ in GENERATION_METHODS])
            raise ValueError(f"未知 only-method：{args.only_method}\n支持的方法：{supported}")

    logger.info("加载样本数：%s", len(samples))
    logger.info("方法数：%s", len(selected_methods))
    logger.info("方法列表：%s", ", ".join([m for m, _ in selected_methods]))

    done_keys = load_done_keys(output_file) if args.resume else set()
    if args.resume:
        logger.info("断点续跑：已完成记录数：%s", len(done_keys))

    client = DeepSeekClient()
    total_tasks = len(samples) * len(selected_methods)
    finished = 0
    skipped = 0
    failed = 0

    for sample_idx, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id", sample_idx))
        if not str(sample.get("sample_id", "")).strip():
            sample["sample_id"] = sample_id

        for method_name, prompt_builder in selected_methods:
            key = (sample_id, method_name)
            if args.resume and key in done_keys:
                skipped += 1
                finished += 1
                continue

            try:
                prompt = prompt_builder(sample)
                row = generate_one(client=client, sample=sample, method_name=method_name, prompt=prompt)
                if row.get("generation_error"):
                    failed += 1
                    logger.warning("生成失败 sample_id=%s method=%s error=%s", sample_id, method_name, row.get("generation_error"))
                append_jsonl(output_file, row)
            except Exception as exc:
                failed += 1
                logger.exception("未预期失败 sample_id=%s method=%s error=%s", sample_id, method_name, exc)
                append_jsonl(output_file, {
                    "sample_id": sample_id,
                    "method": method_name,
                    "query": str(sample.get("query", "")),
                    "generated_answer": "",
                    "response": "",
                    "source_text": get_source_text(sample),
                    "judge_error": "",
                    "generation_error": str(exc),
                })

            finished += 1
            if finished % 10 == 0 or finished == total_tasks:
                logger.info("进度：%s/%s | 跳过=%s | 失败=%s | 输出=%s", finished, total_tasks, skipped, failed, output_file)
            time.sleep(float(REQUEST_INTERVAL))

    logger.info("去层消融生成完成")
    logger.info("输出文件：%s", output_file)
    logger.info("总任务=%s | 跳过=%s | 失败=%s", total_tasks, skipped, failed)

    score_file = DEFAULT_TARGET_DIR / "scores" / "leave_one_out_full23_scores.csv"
    print("\n✅ MRGEO 去层消融生成完成")
    print(f"   输出文件：{output_file}")
    print(f"   总任务：{total_tasks} | 跳过：{skipped} | 失败：{failed}")
    print("\n下一步可运行 23 指标评分：")
    print(
        'python .\\run_evaluation_full23.py '
        f'--input "{output_file}" '
        f'--output "{score_file}" '
        '--max-samples 999999 '
        '--resume'
    )


if __name__ == "__main__":
    main()
