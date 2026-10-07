import json

input_file = "GEO-bench.jsonl"
output_file = "GEO_bench_url_cleaned.txt"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for sample_idx, line in enumerate(f_in, start=1):
        item = json.loads(line)
        sources = item.get("sources", [])

        for doc_idx, source in enumerate(sources, start=1):
            url = str(source.get("url", "")).strip()
            cleaned_text = str(source.get("cleaned_text", "")).strip()

            f_out.write("=" * 60 + "\n")
            f_out.write(f"样本编号: {sample_idx}\n")
            f_out.write(f"文档编号: {doc_idx}\n")
            f_out.write("=" * 60 + "\n")

            f_out.write("URL:\n")
            f_out.write(url + "\n\n")

            f_out.write("CLEANED_TEXT:\n")
            f_out.write(cleaned_text + "\n\n")
            f_out.write("\n")

print(f"整理完成，已保存到: {output_file}")
