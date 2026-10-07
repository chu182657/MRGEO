import json

file_path = "GEO-bench.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    first_line = next(f)
    first_item = json.loads(first_line)

print("第一条数据读取成功！")
print()
print("这条数据有哪些字段：")
print(first_item.keys())
print()
print("query 内容：")
print(first_item["query"])
print()
print("sources 数量：")
print(len(first_item["sources"]))
