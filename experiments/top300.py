import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ====== 你要改的配置 ======
INPUT_FILE = "GEO-bench.jsonl"
OUTPUT_FILE = "short_question_chains_top300.jsonl"
LIMIT = 300
MAX_WORKERS = 6
MAX_RETRY = 2
REQUEST_TIMEOUT = 20

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", os.getenv("API_KEY", "")),
    base_url="https://api.deepseek.com"
)

write_lock = threading.Lock()


def get_topic(item, idx):
    return (
        item.get("big_theme")
        or item.get("theme")
        or item.get("topic")
        or item.get("label")
        or f"sample_{idx}"
    )


def build_prompt(topic):
    return f"""
请围绕主题“{topic}”生成 2 到 4 个连续的问题链。

要求：
1. 问题之间要有递进关系
2. 使用简洁英文
3. 只输出 JSON
4. JSON格式如下：
{{"questions":["Q1","Q2","Q3"]}}

不要输出任何解释。
""".strip()


def call_api(topic):
    prompt = build_prompt(topic)

    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates short sequential question chains."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                timeout=REQUEST_TIMEOUT,
            )

            text = resp.choices[0].message.content.strip()
            data = json.loads(text)
            questions = data.get("questions", [])

            if not isinstance(questions, list):
                raise ValueError("questions is not a list")

            questions = [str(q).strip() for q in questions if str(q).strip()]
            if not (2 <= len(questions) <= 4):
                raise ValueError(f"question count invalid: {len(questions)}")

            return {
                "status": "success",
                "questions": questions,
                "error": ""
            }

        except Exception as e:
            if attempt < MAX_RETRY - 1:
                time.sleep(1.5)
            else:
                return {
                    "status": "failed",
                    "questions": [],
                    "error": str(e)
                }


def process_one(args):
    idx, item = args
    sample_id = item.get("sample_id", idx)
    topic = get_topic(item, idx)

    result = call_api(topic)
    result_obj = {
        "sample_id": sample_id,
        "topic": topic,
        "questions": result["questions"],
        "question_count": len(result["questions"]),
        "status": result["status"],
        "error": result["error"]
    }
    return idx, result_obj


def main():
    items = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx > LIMIT:
                break
            items.append((idx, json.loads(line)))

    print(f"加载完成，共 {len(items)} 条，开始并发生成...")

    done = 0
    start_time = time.time()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_one, x) for x in items]

            for future in as_completed(futures):
                idx, result_obj = future.result()

                with write_lock:
                    f_out.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                    f_out.flush()

                done += 1
                elapsed = time.time() - start_time
                avg = elapsed / done
                remaining = avg * (len(items) - done)

                print(
                    f"[{done}/{len(items)}] sample_id={result_obj['sample_id']} | "
                    f"{result_obj['topic']} | {result_obj['status']} | "
                    f"已用 {elapsed:.1f}s | 预计剩余 {remaining:.1f}s"
                )

    total = time.time() - start_time
    print(f"\n完成，结果已保存到: {OUTPUT_FILE}")
    print(f"总耗时: {total:.1f} 秒")


if __name__ == "__main__":
    main()
