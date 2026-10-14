"""debug_gdc.py — 探测 GDC TCGA-LUAD 有哪些 clinical 文件"""
import requests, json

# 宽查询: 看 TCGA-LUAD 所有 data_type / data_category 有啥
r = requests.post(
    "https://api.gdc.cancer.gov/files",
    json={
        "filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id",
                          "value": ["TCGA-LUAD"]}},
            ]
        },
        "fields": "data_category,data_type,data_format,file_name,file_id",
        "format": "JSON",
        "size": "2000",
    },
    timeout=60
)
data = r.json()
hits = data.get("data", {}).get("hits", [])
print(f"Total TCGA-LUAD files: {len(hits)}")

# 看各 data_type 分布
from collections import Counter
dt_counter = Counter()
dc_counter = Counter()
fmt_counter = Counter()
for h in hits:
    dt_counter[h.get("data_type", "?")] += 1
    dc_counter[h.get("data_category", "?")] += 1
    fmt_counter[h.get("data_format", "?")] += 1

print("\ndata_category:")
for k, v in dc_counter.most_common():
    print(f"  {k}: {v}")

print("\ndata_type:")
for k, v in dt_counter.most_common():
    print(f"  {k}: {v}")

print("\ndata_format:")
for k, v in fmt_counter.most_common():
    print(f"  {k}: {v}")

# 看 clinical 相关的样例
print("\nclinical 相关的前 5 个文件:")
for h in hits:
    name = h.get("file_name", "")
    cat = h.get("data_category", "")
    t = h.get("data_type", "")
    if "clinical" in name.lower() or "Clinical" in cat or "Supplement" in t or "supplement" in t:
        print(f"  file_name={name}")
        print(f"  data_category={cat}, data_type={t}, format={h.get('data_format')}")
        print(f"  file_id={h.get('file_id')}")
        print()
        if name.count("Clinical") + name.count("clinical") >= 2:
            break
