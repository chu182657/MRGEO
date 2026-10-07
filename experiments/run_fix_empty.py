"""
run_fix_empty.py
================
修复生成结果中 generated_answer 为空的记录。
找出所有空结果，重新调用 API 生成，并更新 jsonl 文件。
"""

import json
import logging
import time
from pathlib import Path

from config import OUTPUT_DIR
from data_loader import load_samples
from gemini_client import call_gemini
from methods import build_prompt


def setup_logger():
    logger = logging.getLogger("run_fix_empty")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def main():
    logger = setup_logger()

    output_file = OUTPUT_DIR / "generations" / "generation_results.jsonl"

    if not output_file.exists():
        print("❌ 找不到生成结果文件，请先运行 run_generation.py")
        return

    # ── 第一步：读取所有记录，找出空结果 ──
    all_records = []
    empty_keys = set()   # (sample_id, method)

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                all_records.append(record)
                answer = str(record.get("generated_answer", "")).strip()
                if not answer:
                    key = (str(record["sample_id"]), str(record["method"]))
                    empty_keys.add(key)
            except Exception:
                continue

    logger.info(f"共读取 {len(all_records)} 条记录，其中空结果 {len(empty_keys)} 条")

    if not empty_keys:
        print("✅ 没有空结果，无需修复！")
        return

    # ── 第二步：构建 sample_id -> sample 的映射 ──
    samples = load_samples()
    sample_map = {str(s["sample_id"]): s for s in samples}

    # ── 第三步：重新生成空结果 ──
    fixed = 0
    still_failed = 0
    fix_results = {}   # (sample_id, method) -> new_answer

    for idx, (sample_id, method) in enumerate(empty_keys, 1):
        sample = sample_map.get(sample_id)
        if not sample:
            logger.warning(f"找不到样本 sample_id={sample_id}，跳过")
            still_failed += 1
            continue

        logger.info(f"[{idx}/{len(empty_keys)}] 重新生成 sample_id={sample_id} method={method}")
        try:
            prompt = build_prompt(method, sample)
            answer = call_gemini(prompt)

            if answer.strip():
                fix_results[(sample_id, method)] = answer
                fixed += 1
                logger.info(f"✅ 修复成功 sample_id={sample_id} method={method}")
            else:
                still_failed += 1
                logger.warning(f"❌ 仍然失败 sample_id={sample_id} method={method}")

        except Exception as e:
            still_failed += 1
            logger.error(f"异常 sample_id={sample_id} method={method} error={e}")

        time.sleep(1.0)  # 避免限流

    # ── 第四步：用新结果覆盖原记录，重写文件 ──
    if fix_results:
        new_records = []
        for record in all_records:
            key = (str(record.get("sample_id", "")), str(record.get("method", "")))
            if key in fix_results:
                record["generated_answer"] = fix_results[key]  # 替换为新结果
            new_records.append(record)

        # 备份原文件
        backup_file = output_file.with_suffix(".jsonl.bak")
        output_file.rename(backup_file)
        logger.info(f"原文件已备份为 {backup_file}")

        # 写入新文件
        with output_file.open("w", encoding="utf-8") as f:
            for record in new_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"文件已更新：{output_file}")

    print(f"\n✅ 修复完成！成功修复 {fixed} 条，仍然失败 {still_failed} 条")


if __name__ == "__main__":
    main()