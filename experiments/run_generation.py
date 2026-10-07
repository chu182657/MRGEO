"""
run_generation.py
=================
批量生成回答并保存结果。

用法：
    python run_generation.py          # 全量运行（200条 × 14种方法）
    python run_generation.py --test   # 测试模式（只跑前5条 × 2种方法）
"""

import argparse
import json
import logging
import time
from pathlib import Path

from config import METHOD_LIST, OUTPUT_DIR
from data_loader import load_samples
from gemini_client import call_gemini
from methods import build_prompt


def setup_logger() -> logging.Logger:
    """初始化日志（控制台 + 文件）。"""
    logger = logging.getLogger("run_generation")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir / "generation.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_existing_keys(output_file: Path) -> set:
    """
    读取已有的生成结果，返回已完成的 (sample_id, method) 集合。
    用于断点续跑，跳过已完成的任务。
    """
    done = set()
    if not output_file.exists():
        return done
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = (str(record.get("sample_id", "")), str(record.get("method", "")))
                done.add(key)
            except Exception:
                continue
    return done


def main():
    parser = argparse.ArgumentParser(description="GEO 批量生成脚本")
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式：只处理前5条样本，只跑2种方法"
    )
    args = parser.parse_args()

    logger = setup_logger()

    # 读取样本
    logger.info("正在加载样本数据...")
    samples = load_samples()
    logger.info(f"共加载 {len(samples)} 条样本")

    # 确定方法列表
    methods = METHOD_LIST
    if args.test:
        samples = samples[:5]
        methods = ["statistics", "m2geo"]
        logger.info("【测试模式】只处理前5条样本，方法：statistics, m2geo")
    else:
        # 正式运行：限制前200条
        samples = samples[:200]
        logger.info(f"限制为前200条样本，共 {len(samples)} 条")

    # 输出文件
    output_dir = OUTPUT_DIR / "generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "generation_results.jsonl"

    # 加载已完成的任务（断点续跑）
    done_keys = load_existing_keys(output_file)
    logger.info(f"已有 {len(done_keys)} 条完成记录，将跳过这些任务")

    # 统计
    total = len(samples) * len(methods)
    completed = 0
    skipped = 0
    failed = 0

    logger.info(f"开始生成，共 {total} 个任务（{len(samples)} 条样本 × {len(methods)} 种方法）")

    with output_file.open("a", encoding="utf-8") as f:
        for sample in samples:
            sample_id = str(sample.get("sample_id", ""))
            query = str(sample.get("query", ""))

            for method in methods:
                key = (sample_id, method)

                # 断点续跑：跳过已完成
                if key in done_keys:
                    skipped += 1
                    continue

                try:
                    # 构造 prompt
                    prompt = build_prompt(method, sample)

                    # 调用 API
                    answer = call_gemini(prompt)

                    # 构造结果记录
                    record = {
                        "sample_id": sample_id,
                        "method": method,
                        "query": query,
                        "generated_answer": answer,
                        "total_docs": len(sample.get("documents", [])),
                    }

                    # 写入文件（每条立即写入，防止中断丢失）
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

                    completed += 1

                    # 每完成50条打印进度
                    if completed % 50 == 0:
                        logger.info(
                            f"进度：{completed + skipped}/{total} "
                            f"（完成{completed} 跳过{skipped} 失败{failed}）"
                        )

                    # 避免 API 限流，每次请求后短暂等待
                    time.sleep(0.5)

                except Exception as e:
                    failed += 1
                    logger.error(
                        f"生成失败 sample_id={sample_id} method={method} error={e}"
                    )
                    continue

    logger.info(
        f"生成完成！共完成{completed}条，跳过{skipped}条，失败{failed}条"
    )
    logger.info(f"结果保存在：{output_file}")
    print(f"\n✅ 生成完成！结果保存在：{output_file}")
    print(f"   完成：{completed} | 跳过：{skipped} | 失败：{failed}")


if __name__ == "__main__":
    main()