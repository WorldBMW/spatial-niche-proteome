"""完整重跑 CPTAC-LUAD / CPTAC-LSCC TCIA manifest 查询 + 下载"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
import requests, pandas as pd
from pathlib import Path

OUT = Path(r"e:\ai_paper\experiment2_fm_time\data\raw\CPTAC")
OUT.mkdir(parents=True, exist_ok=True)
TCIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

def api_get(path, **params):
    r = requests.get(f"{TCIA}/{path}", params=params, timeout=120, verify=False)
    r.raise_for_status()
    return r.json()

print("=" * 60)
print("Step 1: 列出 TCIA 所有 collection")
print("=" * 60)
try:
    all_cols = api_get("getCollectionValues")
    print(f"  Total collections: {len(all_cols)}")
    # CPTAC Lung 相关
    cptac_lung = [c for c in all_cols if isinstance(c, str) and "CPTAC" in c and ("LUNG" in c.upper() or "LUAD" in c.upper() or "LSCC" in c.upper())]
    print(f"  CPTAC-Lung: {cptac_lung}")
except Exception as e:
    print(f"  FAIL: {e}")
    cptac_lung = ["CPTAC-LUAD", "CPTAC-LSCC", "CPTAC_NSCLC_LUAD", "CPTAC_NSCLC_LSCC"]

# fallback: 直接硬查几个可能的名字
if not cptac_lung:
    cptac_lung = ["CPTAC-LUAD", "CPTAC-LSCC", "CPTAC_NSCLC", "CPTAC NSCLC"]

print(f"  尝试 collections: {cptac_lung}")

print()
print("=" * 60)
print("Step 2: 逐个查询 series")
print("=" * 60)

all_series = []
for col in cptac_lung:
    print(f"\n  → {col} ... ", end="", flush=True)
    try:
        series = api_get("getSeries", Collection=col, format="JSON")
        print(f"{len(series)} series")
        for s in series:
            mod = s.get("Modality", "")
            if mod in ("SM",) or "pathology" in s.get("ImageType","").lower():
                all_series.append({
                    "SeriesInstanceUID": s.get("SeriesInstanceUID"),
                    "StudyInstanceUID": s.get("StudyInstanceUID"),
                    "PatientID": s.get("PatientID"),
                    "Modality": mod,
                    "BodyPart": s.get("BodyPartExamined", ""),
                    "NumImages": s.get("NumberofImages", 1),
                    "FileSize_MB": round(s.get("FileSize", 0) / 1e6, 1),
                    "Collection": col,
                })
    except Exception as e:
        print(f"FAIL: {e}")

if not all_series:
    print("\n  没找到 CPTAC-Lung series, 换关键词试一下...")
    for kw in ["CPTAC", "LUAD", "Lung", "NSCLC"]:
        print(f"\n  → 搜索: {kw} ... ")
        try:
            series = api_get("getSeries", Collection=kw, format="JSON")
            for s in series[:3]:
                print(f"     Collection={s.get('CollectionName','?')}, Patient={s.get('PatientID','?')}, Mod={s.get('Modality','?')}")
        except Exception as e:
            print(f"    FAIL: {e}")

print(f"\n  总计找到 {len(all_series)} pathology series")

if all_series:
    df = pd.DataFrame(all_series)
    print(f"  Patients: {df['PatientID'].nunique()}")
    total_gb = df["FileSize_MB"].sum() / 1024
    print(f"  估算大小: {total_gb:.1f} GB")
    print(df.head(10).to_string(index=False))
    
    manifest = OUT / "cptac_tcia_manifest.csv"
    df.to_csv(manifest, index=False)
    print(f"\n  manifest 保存: {manifest}")
else:
    print("\n  ❌ TCIA 没找到 CPTAC-Lung WSI. 需要手动查 TCIA website 确认 collection 名称.")

print()
print("=" * 60)
print("Step 3: 检查 GDC TCGA (补充更多 patients)")
print("=" * 60)
# 查一下 TCGA-LUAD/LUSC 在 GDC 有多少 patients 有 WSI
gdc_url = "https://api.gdc.cancer.gov/files"
filters = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "data_category", "value": ["Biospecimen"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Clinical Supplement"]}},
    ]
}
import urllib.parse, urllib.request
params = {"filters": json.dumps(filters), "size": "1", "format": "JSON"}
print(f"  测 GDC files API...")
try:
    req = urllib.request.Request(f"{gdc_url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "curl"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())
    print(f"  GDC files OK: {d.get('data',{}).get('pagination',{}).get('total',0)}")
except Exception as e:
    print(f"  GDC FAIL: {e}")

# 查 TCGA-LUAD + TCGA-LUSC clinical supplement patient 数
filters2 = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LUAD", "TCGA-LUSC"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Clinical Supplement"]}},
    ]
}
params2 = {"filters": json.dumps(filters2), "size": "1", "format": "JSON"}
try:
    req = urllib.request.Request(f"{gdc_url}?{urllib.parse.urlencode(params2)}", headers={"User-Agent": "curl"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())
    total = d.get("data",{}).get("pagination",{}).get("total",0)
    print(f"  TCGA-LUAD+LUSC clinical supplement: {total} files")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n✅ Done. 有 manifest 就启动下载, 没 manifest 就要手动查 TCIA 确认 collection 名字")
