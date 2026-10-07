import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

INPUT_FILE = "group_topic_fast_en.csv"
OUTPUT_FILE = "reddit_questions_no_api.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}

SLEEP_SECONDS = 1
SEARCH_LIMIT = 2
COMMENT_LIMIT = 5


def load_queries(csv_file):
    df = pd.read_csv(csv_file)
    records = []
    for _, row in df.iterrows():
        query = str(row.get("search_query", "")).strip()
        topic_label = str(row.get("topic_label", "")).strip()
        sample_id = row.get("sample_id", "")

        if query:
            records.append({
                "sample_id": sample_id,
                "topic_label": topic_label,
                "search_query": query
            })
    return records


def search_reddit(query, limit=2):
    url = f"https://old.reddit.com/search/?q={quote(query)}&sort=relevance&t=all"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    things = soup.select("div.thing")

    for thing in things:
        a = thing.select_one("a.search-title")
        if not a:
            a = thing.select_one("a.title")
        if not a:
            continue

        title = a.get_text(" ", strip=True)
        post_url = a.get("href", "").strip()

        if not post_url:
            continue

        if post_url.startswith("/"):
            post_url = "https://old.reddit.com" + post_url

        results.append({
            "post_title": title,
            "post_url": post_url
        })

        if len(results) >= limit:
            break

    return results


def to_json_url(post_url):
    post_url = post_url.rstrip("/")
    if "reddit.com" in post_url:
        return post_url + ".json"
    return post_url


def fetch_post_json(post_url):
    json_url = to_json_url(post_url)
    resp = requests.get(json_url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def extract_post_and_comments(post_json, comment_limit=5):
    title = ""
    selftext = ""
    subreddit = ""
    comments = []

    try:
        post_data = post_json[0]["data"]["children"][0]["data"]
        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")
        subreddit = post_data.get("subreddit", "")
    except:
        pass

    try:
        comment_nodes = post_json[1]["data"]["children"]
        for node in comment_nodes:
            if node.get("kind") != "t1":
                continue
            body = node.get("data", {}).get("body", "").strip()
            if body:
                comments.append(body)
            if len(comments) >= comment_limit:
                break
    except:
        pass

    return title, selftext, subreddit, comments


def extract_questions(text):
    if not text:
        return []

    text = text.replace("\n", " ").strip()
    candidates = re.findall(r'[^.!?\n]*\?+', text)

    cleaned = []
    for c in candidates:
        q = c.strip()
        q = re.sub(r"\s+", " ", q)
        if len(q) < 15 or len(q) > 200:
            continue
        cleaned.append(q)

    seen = set()
    result = []
    for q in cleaned:
        if q not in seen:
            seen.add(q)
            result.append(q)

    return result


def main():
    query_records = load_queries(INPUT_FILE)
    rows = []

    for idx, item in enumerate(query_records, start=1):
        sample_id = item["sample_id"]
        topic_label = item["topic_label"]
        search_query = item["search_query"]

        print(f"[{idx}/{len(query_records)}] Searching Reddit for: {search_query}")

        try:
            posts = search_reddit(search_query, limit=SEARCH_LIMIT)
        except Exception as e:
            rows.append({
                "sample_id": sample_id,
                "topic_label": topic_label,
                "search_query": search_query,
                "source_site": "reddit",
                "post_title": "",
                "post_url": "",
                "subreddit": "",
                "selftext": "",
                "comment_texts": "",
                "extracted_questions": "",
                "error": f"search_error: {e}"
            })
            continue

        for post in posts:
            post_title = post["post_title"]
            post_url = post["post_url"]

            try:
                post_json = fetch_post_json(post_url)
                title, selftext, subreddit, comments = extract_post_and_comments(
                    post_json, comment_limit=COMMENT_LIMIT
                )

                combined_text = " ".join([title, selftext] + comments)
                questions = extract_questions(combined_text)

                rows.append({
                    "sample_id": sample_id,
                    "topic_label": topic_label,
                    "search_query": search_query,
                    "source_site": "reddit",
                    "post_title": title if title else post_title,
                    "post_url": post_url,
                    "subreddit": subreddit,
                    "selftext": selftext,
                    "comment_texts": " || ".join(comments),
                    "extracted_questions": " || ".join(questions),
                    "error": ""
                })

            except Exception as e:
                rows.append({
                    "sample_id": sample_id,
                    "topic_label": topic_label,
                    "search_query": search_query,
                    "source_site": "reddit",
                    "post_title": post_title,
                    "post_url": post_url,
                    "subreddit": "",
                    "selftext": "",
                    "comment_texts": "",
                    "extracted_questions": "",
                    "error": f"post_error: {e}"
                })

            time.sleep(SLEEP_SECONDS)

        if idx % 5 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Done! Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
