import os
import json
import time
import pandas as pd
from openai import OpenAI

INPUT_FILE = "GEO-bench.jsonl"
OUTPUT_FILE = "group_topic_fast_en.csv"
MODEL_NAME = "deepseek-chat"

# 为了加速，缩短输入
MAX_CHARS = 2000

# 测试用 10，正式跑改成 None
LIMIT = None

# 如果不报限流，可以设为 0
SLEEP_SECONDS = 0

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", os.getenv("API_KEY", "")),
    base_url="https://api.deepseek.com"
)


def build_prompt(text):
    return f"""
Read the articles and output:
1. one broad English topic label
2. one simple English search query for recommendation-style discussions

Return JSON only:
{{
  "topic_label": "...",
  "search_query": "..."
}}

Rules:
- English only
- topic_label should be broad and useful, like sports, sneakers, tourism, traditional festivals, skin care
- do not summarize
- keep it short

Articles:
{text}
""".strip()


def call_model(merged_text):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=50,
        messages=[
            {
                "role": "system",
                "content": "You map grouped documents to short English topic labels."
            },
            {
                "role": "user",
                "content": build_prompt(merged_text)
            }
        ]
    )
    return response.choices[0].message.content.strip()


def parse_json_result(result_text):
    txt = result_text.strip()
    try:
        return json.loads(txt)
    except:
        pass

    if txt.startswith("```"):
        txt = txt.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(txt)
        except:
            pass

    return {
        "topic_label": "",
        "search_query": "",
        "raw_output": result_text
    }


def normalize_topic_label(label):
    if not isinstance(label, str):
        return ""
    label = label.strip().lower()

    mapping = {
        "sport": "sports",
        "festival": "traditional festivals",
        "festivals": "traditional festivals",
        "shoe": "sneakers",
        "shoes": "sneakers",
        "travel": "tourism",
        "skincare": "skin care"
    }
    return mapping.get(label, label)


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

        merged_text = "\n\n".join(texts)[:MAX_CHARS]

        try:
            result_text = call_model(merged_text)
            parsed = parse_json_result(result_text)

            topic_label = normalize_topic_label(parsed.get("topic_label", ""))
            search_query = parsed.get("search_query", "")
            error_msg = ""

        except Exception as e:
            topic_label = ""
            search_query = ""
            result_text = ""
            error_msg = str(e)

        rows.append({
            "sample_id": sample_idx,
            "doc_count": len(sources),
            "topic_label": topic_label,
            "search_query": search_query,
            "urls": " | ".join(urls),
            "raw_model_output": result_text,
            "error": error_msg
        })

        if sample_idx % 20 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        print(f"Processed sample {sample_idx}: {topic_label}")
        time.sleep(SLEEP_SECONDS)

pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Done! Results saved to: {OUTPUT_FILE}")
