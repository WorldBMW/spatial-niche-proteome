"""debug_clinical.py — 看 GDC Clinical 里到底有什么"""
import requests

r = requests.post(
    "https://api.gdc.cancer.gov/files",
    json={
        "filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id",
                          "value": ["TCGA-LUAD", "TCGA-LUSC"]}},
                {"op": "in", "content": {"field": "data_category",
                          "value": ["Clinical"]}},
            ]
        },
        "fields": "file_name,data_category,data_type,data_format,file_id",
        "format": "JSON",
        "size": "2227",
    },
    timeout=120
)

hits = r.json()["data"]["hits"]
print(f"Total Clinical files: {len(hits)}")

# 按 data_type 分组
from collections import Counter, defaultdict
by_type = defaultdict(list)
for h in hits:
    by_type[h["data_type"]].append(h)

for dt, files in by_type.items():
    formats = Counter(f["data_format"] for f in files)
    print(f"\n{dt} ({len(files)} files):")
    print(f"  formats: {dict(formats)}")
    # 看一个样例的 file_name
    for f in files[:2]:
        print(f"  sample: {f['file_name']}")
