"""
run_generation_ablation.py
==========================

MRGEOA / M²GEO 逐层消融实验生成脚本。

用途：
    生成“原始文章 Raw + 逐层加工文章”的消融实验结果。

特点：
    1. 不修改原有 run_generation.py
    2. 不覆盖原有 generation_results.jsonl
    3. 单独输出 ablation_generation_results.jsonl
    4. 支持 --resume 断点续跑
    5. 每条样本生成多个阶段版本：
        - MRGEOA-Raw
        - MRGEOA-S1
        - MRGEOA-S1S2
        - MRGEOA-S1S2S3
        - MRGEOA-S1S2S3S4
        - MRGEOA-Full

默认输入：
    当前项目的 data/samples.json

默认输出：
    当前用户桌面：
    Desktop/222/outputs/generations/ablation_generation_results.jsonl

运行示例：
    先测试 3 条：
        python run_generation_ablation.py --max-samples 3 --resume

    全量运行：
        python run_generation_ablation.py --resume

    指定输出：
        python run_generation_ablation.py --output "C:\\Users\\你的用户名\\Desktop\\222\\outputs\\generations\\ablation_generation_results.jsonl" --resume

后续 23 指标评分：
    python run_evaluation_full23.py ^
      --input "$HOME\\Desktop\\222\\outputs\\generations\\ablation_generation_results.jsonl" ^
      --output "$HOME\\Desktop\\222\\outputs\\scores\\ablation_full23_scores.csv" ^
      --max-samples 999999 ^
      --resume
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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

# 默认输出到桌面 222 文件夹，不覆盖 111 原实验结果
DEFAULT_TARGET_DIR = OUTPUT_DIR
DEFAULT_OUTPUT_FILE = DEFAULT_TARGET_DIR / "generations" / "ablation_generation_results.jsonl"
DEFAULT_LOG_FILE = DEFAULT_TARGET_DIR / "logs" / "generation_ablation.log"


# ============================================================
# 日志
# ============================================================

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("run_generation_ablation")

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
    """
    从 sample 中提取 query 和 documents。
    兼容 documents 为 list[str] 或 list[dict] 的情况。
    """
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
    """
    将参考文档格式化为 prompt 中使用的 [Doc X] 格式。
    """
    if not docs:
        return "（无可用参考资料）"

    parts = []
    for idx, doc in enumerate(docs, start=1):
        parts.append(f"[Doc {idx}]\n{doc}")

    return "\n\n".join(parts)


def get_source_text(sample: Dict[str, Any]) -> str:
    """
    保存 source_text，方便后续 23 指标评分时 judge_full.py 使用。
    """
    _, docs = extract_fields(sample)
    return format_docs(docs)


# ============================================================
# Prompt 构造：Raw + 逐层消融
# ============================================================

def build_raw_prompt(sample: Dict[str, Any]) -> str:
    """
    Raw：不使用 MRGEOA 加工策略，只让模型基于参考资料直接写文章。
    """
    query, docs = extract_fields(sample)
    formatted = format_docs(docs)

    return f"""你是一位信息整合写作助手。

请基于以下参考资料，直接回答用户问题。

【参考资料】
{formatted}

【用户问题】
{query}

【写作要求】
1. 只基于参考资料作答，不要编造资料中不存在的事实。
2. 文章应自然、清楚、完整。
3. 不需要执行任何额外优化流程。
4. 回答长度约 200-400 字。

请直接输出最终文章，不要解释你的写作过程。"""


def build_s1_prompt(sample: Dict[str, Any]) -> str:
    """
    S1：内容准入 Content Admission。
    只筛选与问题直接相关的内容，再生成文章。
    """
    query, docs = extract_fields(sample)
    formatted = format_docs(docs)

    return f"""你是一个问答优化系统，现在只执行 MRGEOA 的第 1 层流程：内容准入（Content Admission）。

【参考资料】
{formatted}

【用户问题】
{query}

【第 1 层：内容准入】
任务：从所有参考资料中筛选出与用户问题直接相关的内容片段。
要求：
1. 识别每个文档中与问题直接相关的关键事实、关键句子或关键段落。
2. 忽略与问题无关或弱相关的信息。
3. 最终文章只能使用筛选后的相关信息。
4. 不要编造资料中不存在的事实。

【最终文章要求】
基于第 1 层筛选结果，写出一篇自然、清楚、完整的回答，长度约 200-400 字。

请直接输出最终文章，不要输出分析过程。"""


def build_s1s2_prompt(sample: Dict[str, Any]) -> str:
    """
    S1+S2：内容准入 + 生成偏好。
    """
    query, docs = extract_fields(sample)
    formatted = format_docs(docs)

    return f"""你是一个问答优化系统，现在执行 MRGEOA 的前 2 层流程。

【参考资料】
{formatted}

【用户问题】
{query}

【第 1 层：内容准入 Content Admission】
从参考资料中筛选出与用户问题直接相关的内容，忽略无关信息。

【第 2 层：生成偏好 Generation Preference】
对筛选出的内容进行优先级排序：
- 优先级 A：能直接回答用户问题的核心信息。
- 优先级 B：支撑核心信息的背景、数据、例子或说明。
- 优先级 C：相关但非必要的补充信息。

【最终文章要求】
1. 最终文章应优先呈现优先级 A 和 B 的内容。
2. 内容必须基于参考资料。
3. 不要编造资料中不存在的事实。
4. 文章结构自然清楚，长度约 200-400 字。

请直接输出最终文章，不要输出分析过程。"""


def build_s1s2s3_prompt(sample: Dict[str, Any]) -> str:
    """
    S1+S2+S3：内容准入 + 生成偏好 + 语义稳定。
    """
    query, docs = extract_fields(sample)
    formatted = format_docs(docs)

    return f"""你是一个问答优化系统，现在执行 MRGEOA 的前 3 层流程。

【参考资料】
{formatted}

【用户问题】
{query}

【第 1 层：内容准入 Content Admission】
筛选出与用户问题直接相关的内容，剔除无关信息。

【第 2 层：生成偏好 Generation Preference】
按照重要性排序信息：
- 优先级 A：直接回答问题的核心信息。
- 优先级 B：支撑核心信息的背景和证据。
- 优先级 C：相关补充信息。

【第 3 层：语义稳定 Semantic Stability】
检查并保证：
1. 最终文章始终围绕用户问题的核心关键词。
2. 不引入偏离主题的内容。
3. 不改变参考资料的原始含义。
4. 不夸大、不歪曲、不遗漏关键限制。

【最终文章要求】
基于前三层处理结果，写出语义稳定、主题集中、忠实可靠的文章，长度约 200-400 字。

请直接输出最终文章，不要输出分析过程。"""


def build_s1s2s3s4_prompt(sample: Dict[str, Any]) -> str:
    """
    S1+S2+S3+S4：内容准入 + 生成偏好 + 语义稳定 + 多维/结构化增强。
    """
    query, docs = extract_fields(sample)
    formatted = format_docs(docs)

    return f"""你是一个问答优化系统，现在执行 MRGEOA 的前 4 层流程。

【参考资料】
{formatted}

【用户问题】
{query}

【第 1 层：内容准入 Content Admission】
筛选出与用户问题直接相关的内容。

【第 2 层：生成偏好 Generation Preference】
优先使用能直接回答问题的核心信息，其次使用支撑信息和补充信息。

【第 3 层：语义稳定 Semantic Stability】
保证文章始终围绕用户问题，不偏题，不歪曲原文含义。

【第 4 层：多维/结构化增强 Multimodal or Structured Enhancement】
识别并优先利用参考资料中的结构化信息：
- 数字、比例、时间、地点等客观数据。
- 列表、分类、层次结构。
- 因果关系、对比关系、背景关系。
- 重要事实与关键观点。

【最终文章要求】
1. 在忠实资料的基础上，让文章更有信息密度和结构感。
2. 能使用结构化信息时，应自然融入文章。
3. 不要编造资料中不存在的事实。
4. 长度约 250-450 字。

请直接输出最终文章，不要输出分析过程。"""


def build_full_prompt(sample: Dict[str, Any]) -> str:
    """
    Full：完整 MRGEOA / M²GEO 五层流程。
    与 methods.py 中 m2geo 的逻辑一致，但这里要求直接输出最终文章，
    方便消融实验统一评分。
    """
    query, docs = extract_fields(sample)
    formatted = format_docs(docs)

    return f"""你是一个高级问答优化系统，现在执行完整 MRGEOA 五层流程。

【参考资料】
{formatted}

【用户问题】
{query}

【第 1 层：内容准入 Content Admission】
从所有参考资料中筛选出与问题直接相关的内容片段，剔除无关信息。

【第 2 层：生成偏好 Generation Preference】
对筛选出的内容进行优先级排序：
- 优先级 A：直接回答问题的核心信息。
- 优先级 B：支撑核心信息的背景、数据、证据或例子。
- 优先级 C：相关但非必要的补充信息。

【第 3 层：语义稳定 Semantic Stability】
确保最终文章始终围绕用户问题的核心意图，不偏题、不歪曲、不夸大、不引入资料外事实。

【第 4 层：多维/结构化增强 Multimodal or Structured Enhancement】
优先识别并使用资料中的数字、列表、分类、因果关系、时间信息、对比关系和关键事实，提高文章的信息密度与结构清晰度。

【第 5 层：意图对齐 Intent Alignment】
根据用户问题判断用户真正想要获得的信息，并让最终文章直接回应这一意图。
要求：
1. 开头应直接回答核心问题。
2. 中间补充关键依据、背景或结构化信息。
3. 结尾简要总结，不要空泛。
4. 全文必须忠实于参考资料。
5. 长度约 250-450 字。

请只输出最终文章，不要输出五层分析过程。"""


ABLATION_METHODS = [
    ("MRGEOA-Raw", build_raw_prompt),
    ("MRGEOA-S1", build_s1_prompt),
    ("MRGEOA-S1S2", build_s1s2_prompt),
    ("MRGEOA-S1S2S3", build_s1s2s3_prompt),
    ("MRGEOA-S1S2S3S4", build_s1s2s3s4_prompt),
    ("MRGEOA-Full", build_full_prompt),
]


# ============================================================
# 断点续跑
# ============================================================

def load_done_keys(output_file: Path) -> Set[Tuple[str, str]]:
    """
    读取已经完成的 sample_id + method，用于断点续跑。
    """
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
            answer = str(
                item.get("generated_answer")
                or item.get("response")
                or ""
            ).strip()

            if sample_id and method and answer:
                done.add((sample_id, method))

    return done


def append_jsonl(output_file: Path, row: Dict[str, Any]) -> None:
    """
    追加写入一条 JSONL，确保每完成一次调用就保存一次，降低中断损失。
    """
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
    """
    调用 DeepSeek 生成一条消融结果。
    """
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
    parser = argparse.ArgumentParser(
        description="MRGEOA / M²GEO 逐层消融实验生成脚本"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="消融生成结果 JSONL 输出路径",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=999999,
        help="最多运行多少个样本，默认近似全量",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：跳过已经成功生成的 sample_id + method",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有输出文件。不建议使用，除非你确定要重跑。",
    )

    parser.add_argument(
        "--only-method",
        type=str,
        default="",
        help="只运行某一个消融方法，例如 MRGEOA-Raw 或 MRGEOA-Full。默认运行全部。",
    )

    args = parser.parse_args()

    output_file = Path(args.output)
    log_file = DEFAULT_LOG_FILE
    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("MRGEOA / M²GEO 逐层消融实验生成开始")
    logger.info("项目目录：%s", PROJECT_DIR)
    logger.info("输出文件：%s", output_file)

    if output_file.exists() and not args.resume and not args.overwrite:
        raise RuntimeError(
            f"输出文件已存在：{output_file}\n"
            f"为防止覆盖已有结果，程序已停止。\n"
            f"如果要断点续跑，请使用：--resume\n"
            f"如果确定要重跑并覆盖，请使用：--overwrite"
        )

    if output_file.exists() and args.overwrite:
        logger.warning("你使用了 --overwrite，已有输出文件将被删除：%s", output_file)
        output_file.unlink()

    samples = load_samples()
    samples = samples[: max(0, int(args.max_samples))]

    logger.info("加载样本数：%s", len(samples))
    logger.info("消融方法数：%s", len(ABLATION_METHODS))
    logger.info("消融方法：%s", ", ".join([m for m, _ in ABLATION_METHODS]))

    selected_methods = ABLATION_METHODS
    if args.only_method.strip():
        selected_methods = [
            item for item in ABLATION_METHODS
            if item[0] == args.only_method.strip()
        ]
        if not selected_methods:
            supported = ", ".join([m for m, _ in ABLATION_METHODS])
            raise ValueError(
                f"未知 only-method：{args.only_method}\n"
                f"支持的方法：{supported}"
            )

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

        for method_name, prompt_builder in selected_methods:
            key = (sample_id, method_name)

            if args.resume and key in done_keys:
                skipped += 1
                finished += 1
                continue

            try:
                prompt = prompt_builder(sample)

                row = generate_one(
                    client=client,
                    sample=sample,
                    method_name=method_name,
                    prompt=prompt,
                )

                if row.get("generation_error"):
                    failed += 1
                    logger.warning(
                        "生成失败 sample_id=%s method=%s error=%s",
                        sample_id,
                        method_name,
                        row.get("generation_error"),
                    )

                append_jsonl(output_file, row)

            except Exception as exc:
                failed += 1
                logger.exception(
                    "未预期失败 sample_id=%s method=%s error=%s",
                    sample_id,
                    method_name,
                    exc,
                )

                error_row = {
                    "sample_id": sample_id,
                    "method": method_name,
                    "query": str(sample.get("query", "")),
                    "generated_answer": "",
                    "response": "",
                    "source_text": get_source_text(sample),
                    "judge_error": "",
                    "generation_error": str(exc),
                }
                append_jsonl(output_file, error_row)

            finished += 1

            if finished % 10 == 0 or finished == total_tasks:
                logger.info(
                    "进度：%s/%s | 跳过=%s | 失败=%s | 输出=%s",
                    finished,
                    total_tasks,
                    skipped,
                    failed,
                    output_file,
                )

            time.sleep(float(REQUEST_INTERVAL))

    logger.info("消融生成完成")
    logger.info("输出文件：%s", output_file)
    logger.info("总任务=%s | 跳过=%s | 失败=%s", total_tasks, skipped, failed)

    print("\n✅ MRGEOA / M²GEO 消融生成完成")
    print(f"   输出文件：{output_file}")
    print(f"   总任务：{total_tasks} | 跳过：{skipped} | 失败：{failed}")
    print("\n下一步可运行 23 指标评分：")
    print(
        'python .\\run_evaluation_full23.py '
        f'--input "{output_file}" '
        f'--output "{DEFAULT_TARGET_DIR / "scores" / "ablation_full23_scores.csv"}" '
        '--max-samples 999999 '
        '--resume'
    )


if __name__ == "__main__":
    main()
