import os
import re
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# =========================================
# Config
# =========================================
TOPIC_CSV = "group_topic_fast_en.csv"
DOC_JSONL = "GEO-bench.jsonl"

OUTPUT_JSONL = "short_question_chains.jsonl"
OUTPUT_CSV = "short_question_chains.csv"

MAX_ARTICLES_PER_TOPIC = 5
MAX_TITLE_PER_TOPIC = 5

RETRY_TIMES = 2
REQUEST_TIMEOUT = 60
SLEEP_SECONDS = 0.3

if os.path.exists(".env.txt"):
    load_dotenv(".env.txt")
else:
    load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env or .env.txt")


# =========================================
# Helpers
# =========================================
def safe_str(x):
    if x is None:
        return ""
    return str(x)


def normalize_text(text):
    text = safe_str(text).replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] jsonl line {line_no} parse error: {e}")
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_urls_field(urls_value):
    raw = safe_str(urls_value).strip()
    if not raw:
        return []
    if "|" in raw:
        return [u.strip() for u in raw.split("|") if u.strip()]
    return [raw]


def build_doc_index(doc_rows):
    idx_map = {}
    for i, row in enumerate(doc_rows, start=1):
        sid = row.get("sample_id", i)
        try:
            sid = int(sid)
        except:
            sid = i
        idx_map[sid] = row
    return idx_map


def extract_article_titles(doc_row, max_articles=5):
    titles = []
    sources = doc_row.get("sources", [])
    for i, src in enumerate(sources[:max_articles], start=1):
        title = normalize_text(src.get("title", "") or f"Article {i}")
        if title:
            titles.append(title)
    return titles


# =========================================
# API
# =========================================
def call_deepseek(messages, temperature=0.4, max_tokens=300):
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    last_error = None
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            last_error = str(e)

        print(f"[WARN] API failed attempt {attempt}/{RETRY_TIMES}: {last_error}")
        time.sleep(attempt)

    raise RuntimeError(last_error or "DeepSeek API error")


def parse_json_response(text):
    text = text.strip()

    try:
        return json.loads(text)
    except:
        pass

    if "```json" in text:
        try:
            body = text.split("```json", 1)[1].split("```", 1)[0].strip()
            return json.loads(body)
        except:
            pass

    if "```" in text:
        try:
            body = text.split("```", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(body)
        except:
            pass

    raise ValueError("Model output is not valid JSON")


# =========================================
# Prompt
# =========================================
def build_short_chain_prompt(sample_id, topic_label, search_query, article_titles):
    titles_text = "\n".join([f"- {t}" for t in article_titles]) if article_titles else "- None"

    system_prompt = (
        "You generate short conversational question chains for search topics. "
        "Return JSON only."
    )

    user_prompt = f"""
Given:
sample_id: {sample_id}
topic_label: {topic_label}
search_query: {search_query}

Related article titles:
{titles_text}

Task:
Generate one short, natural multi-turn question chain for this topic.

Requirements:
1. Generate ONLY 2 to 4 questions.
2. Questions must be short, natural, and logically connected.
3. The first question should be broad.
4. Later questions can refine, compare, or expand.
5. Keep the output concise.

Return JSON only:
{{
  "sample_id": {sample_id},
  "big_theme": "<short theme>",
  "questions": [
    "question 1",
    "question 2"
  ]
}}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def generate_short_chain(sample_id, topic_label, search_query, article_titles):
    messages = build_short_chain_prompt(sample_id, topic_label, search_query, article_titles)
    raw = call_deepseek(messages, temperature=0.4, max_tokens=250)
    parsed = parse_json_response(raw)

    questions = parsed.get("questions", [])
    questions = [normalize_text(q) for q in questions if normalize_text(q)]

    # 保底修正：只保留2~4个
    if len(questions) > 4:
        questions = questions[:4]

    if len(questions) < 2:
        raise ValueError("Generated questions less than 2")

    return {
        "sample_id": sample_id,
        "big_theme": normalize_text(parsed.get("big_theme", topic_label)),
        "questions": questions
    }


# =========================================
# Main
# =========================================
def main():
    topic_df = pd.read_csv(TOPIC_CSV)
    doc_rows = load_jsonl(DOC_JSONL)
    doc_index = build_doc_index(doc_rows)

    all_results = []

    for idx, row in topic_df.iterrows():
        sample_id = row.get("sample_id", idx + 1)
        try:
            sample_id = int(sample_id)
        except:
            sample_id = idx + 1

        topic_label = normalize_text(row.get("topic_label", ""))
        search_query = normalize_text(row.get("search_query", ""))
        urls = parse_urls_field(row.get("urls", ""))

        print(f"[{idx+1}/{len(topic_df)}] sample_id={sample_id} | {topic_label}")

        doc_row = doc_index.get(sample_id)
        if not doc_row:
            result = {
                "sample_id": sample_id,
                "topic_label": topic_label,
                "search_query": search_query,
                "urls": urls,
                "status": "missing_doc",
                "error": "No matching row in GEO-bench.jsonl"
            }
            all_results.append(result)
            write_jsonl(OUTPUT_JSONL, all_results)
            continue

        article_titles = extract_article_titles(doc_row, max_articles=MAX_ARTICLES_PER_TOPIC)

        try:
            chain = generate_short_chain(
                sample_id=sample_id,
                topic_label=topic_label,
                search_query=search_query,
                article_titles=article_titles
            )

            result = {
                "sample_id": sample_id,
                "topic_label": topic_label,
                "search_query": search_query,
                "urls": urls,
                "article_titles": article_titles,
                "big_theme": chain["big_theme"],
                "question_count": len(chain["questions"]),
                "questions": chain["questions"],
                "status": "ok",
                "error": ""
            }

            all_results.append(result)
            write_jsonl(OUTPUT_JSONL, all_results)

        except Exception as e:
            result = {
                "sample_id": sample_id,
                "topic_label": topic_label,
                "search_query": search_query,
                "urls": urls,
                "article_titles": article_titles,
                "status": "failed",
                "error": str(e)
            }
            all_results.append(result)
            write_jsonl(OUTPUT_JSONL, all_results)

        time.sleep(SLEEP_SECONDS)

    # 转 CSV
    flat_rows = []
    for item in all_results:
        if item.get("status") != "ok":
            flat_rows.append({
                "sample_id": item.get("sample_id", ""),
                "topic_label": item.get("topic_label", ""),
                "search_query": item.get("search_query", ""),
                "big_theme": "",
                "question_count": "",
                "q1": "",
                "q2": "",
                "q3": "",
                "q4": "",
                "status": item.get("status", ""),
                "error": item.get("error", "")
            })
            continue

        qs = item.get("questions", [])
        flat_rows.append({
            "sample_id": item.get("sample_id", ""),
            "topic_label": item.get("topic_label", ""),
            "search_query": item.get("search_query", ""),
            "big_theme": item.get("big_theme", ""),
            "question_count": item.get("question_count", 0),
            "q1": qs[0] if len(qs) > 0 else "",
            "q2": qs[1] if len(qs) > 1 else "",
            "q3": qs[2] if len(qs) > 2 else "",
            "q4": qs[3] if len(qs) > 3 else "",
            "status": item.get("status", ""),
            "error": item.get("error", "")
        })

    pd.DataFrame(flat_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Saved JSONL => {OUTPUT_JSONL}")
    print(f"Saved CSV   => {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
