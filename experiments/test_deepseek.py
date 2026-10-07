import os
import json
import time
import pandas as pd
from openai import OpenAI

# ===== 配置 =====
INPUT_FILE = "GEO-bench.jsonl"
OUTPUT_FILE = "group_keywords_deepseek.csv"
MODEL_NAME = "deepseek-chat"

# 为了控制成本和长度，每组最多取前这么多字符
MAX_CHARS = 8000

# 先测试时可以改成 3、10；全部跑就改成 None
LIMIT = 3

# 每次请求间隔，避免太快
SLEEP_SECONDS = 1


client = OpenAI(
    api_key="把你的新DeepSeek_API_Key写在这里",
    base_url="https://api.deepseek.com"
)


def build_prompt(text):
    return f"""
You are extracting searchable topic keywords from a group of related articles.

Task:
Identify from the articles:
1. one core searchable keyword or phrase
2. 3 to 5 candidate keywords or phrases
3. 3 forum-style search queries

Rules:
- Do NOT summarize the articles.
- Do NOT explain.
- Output concise searchable keywords/phrases only.
- Prefer concrete topics, entities, practices, products, diseases, events, named concepts.
- Avoid overly broad words like sports, culture, health, technology, lifestyle, medicine unless necessary.
- The core keyword should be the best single term someone would use to search discussions about this topic.

Return JSON only in this format:
{{
  "core_keyword": "...",
  "candidate_keywords": ["...", "...", "..."],
  "forum_queries": ["...", "...", "..."]
}}

Articles:
{text}
""".strip()


def call_model(text):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You extract concrete searchable topic keywords from grouped documents."
            },
            {
                "role": "user",
                "content": build_prompt(text)
            }
        ]
    )
    return response.choices[0].message.content.strip()


def parse_json_result(result_text):
    result_text = result_text.strip()

    # 尝试直接解析
    try:
        return json.loads(result_text)
    except:
        pass

    # 如果模型返回了 ```json ... ```，做简单清洗
    if result_text.startswith("```"):
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(result_text)
        except:
            pass

    # 解析失败时，返回兜底结构
    return {
        "core_keyword": "",
        "candidate_keywords": [],
        "forum_queries": [],
        "raw_output": result_text
    }


rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for sample_idx, line in enumerate(f, start=1):
        if LIMIT is not None and sample_idx > LIMIT:
            break

        item = json.loads(line)
        sources = item.get("sources", [])

        texts = []
        urls = []

        for source in sources:
            cleaned_text = str(source.get("cleaned_text", "")).strip()
            url = str(source.get("url", "")).strip()

            if cleaned_text:
                texts.append(cleaned_text)
            if url:
                urls.append(url)

        merged_text = "\n\n".join(texts)
        merged_text = merged_text[:MAX_CHARS]

        try:
            result_text = call_model(merged_text)
            parsed = parse_json_result(result_text)

            core_keyword = parsed.get("core_keyword", "")
            candidate_keywords = parsed.get("candidate_keywords", [])
            forum_queries = parsed.get("forum_queries", [])

            if isinstance(candidate_keywords, list):
                candidate_keywords = " | ".join(candidate_keywords)
            else:
                candidate_keywords = str(candidate_keywords)

            if isinstance(forum_queries, list):
                forum_queries = " | ".join(forum_queries)
            else:
                forum_queries = str(forum_queries)

            error_msg = ""

        except Exception as e:
            core_keyword = ""
            candidate_keywords = ""
            forum_queries = ""
            error_msg = str(e)
            result_text = ""

        rows.append({
            "sample_id": sample_idx,
            "doc_count": len(sources),
            "core_keyword": core_keyword,
            "candidate_keywords": candidate_keywords,
            "forum_queries": forum_queries,
            "urls": " | ".join(urls),
            "raw_model_output": result_text,
            "error": error_msg
        })

        print(f"已处理样本 {sample_idx}")
        time.sleep(SLEEP_SECONDS)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"完成！结果已保存到：{OUTPUT_FILE}")
