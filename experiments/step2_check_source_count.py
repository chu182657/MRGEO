import json

file_path = "GEO-bench.jsonl"

total = 0
count_5 = 0
not_5_items = []

with open(file_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, start=1):
        item = json.loads(line)
        source_count = len(item.get("sources", []))

        total += 1

        if source_count == 5:
            count_5 += 1
        else:
            not_5_items.append({
                "line": idx,
                "query": item.get("query", ""),
                "source_count": source_count
            })

print("检查完成！")
print(f"总样本数: {total}")
print(f"sources 数量等于 5 的样本数: {count_5}")
print(f"sources 数量不等于 5 的样本数: {len(not_5_items)}")
print()

if not_5_items:
    print("下面是不等于 5 的前 10 条样本：")
    for x in not_5_items[:10]:
        print("-" * 50)
        print(f"行号: {x['line']}")
        print(f"source_count: {x['source_count']}")
        print(f"query: {x['query']}")
else:
    print("所有样本的 sources 数量都等于 5。")

