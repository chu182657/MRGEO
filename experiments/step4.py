import json

input_file = "GEO-bench.jsonl"
output_file = "GEO_bench_grouped_sources.txt"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for sample_idx, line in enumerate(f_in, start=1):
        item = json.loads(line)
        sources = item.get("sources", [])

        f_out.write("=" * 60 + "\n")
        f_out.write(f"样本编号: {sample_idx}\n")
        f_out.write(f"文档数量: {len(sources)}\n")
        f_out.write("=" * 60 + "\n\n")

        for doc_idx, source in enumerate(sources, start=1):
            f_out.write(f"[文档 {doc_idx}]\n")
            f_out.write(str(source).strip() + "\n\n")

        f_out.write("\n\n")

print(f"整理完成，已保存到: {output_file}")
