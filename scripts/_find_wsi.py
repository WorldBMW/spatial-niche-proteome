"""确认 CPTAC 病理 WSI 资源位置 + GDC TCGA slide 统计"""
import json, requests, urllib.request, urllib.parse, warnings
warnings.filterwarnings("ignore")

# 1. CPTAC 病理 WSI — 查 cptacportal 或 GDC
print("=" * 60)
print("1. CPTAC 病理 WSI 在哪里?")
print("=" * 60)

# 查 GDC portal: CPTAC project_id = CPTAC-LUAD, CPTAC-LSCC
# 但 CPTAC 的病理 slide 可能走 TCGA 归档 (因为 CPTAC 是 TCGA 的后续)
# 先查 GDC files API: data_category = Biospecimen, data_type = Slide Image
gdc = "https://api.gdc.cancer.gov"

filters_cptac = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["CPTAC-LUAD", "CPTAC-LSCC"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
    ]
}
params = {"filters": json.dumps(filters_cptac), "size": "1", "format": "JSON"}
url = f"{gdc}/files?{urllib.parse.urlencode(params)}"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "curl"})
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read())
    total = d.get("data",{}).get("pagination",{}).get("total",0)
    print(f"  CPTAC-LUAD+LSCC Slide Image (GDC): {total}")
except Exception as e:
    print(f"  FAIL: {e}")

# 查 TCIA CPTAC collections 全列表
print()
print("  TCIA 上有 CPTAC 的哪些 collection (完整 list)")
try:
    cols = requests.get("https://services.cancerimagingarchive.net/nbia-api/services/v1/getCollectionValues", timeout=15, verify=False).json()
    cptac_all = [c for c in cols if "CPTAC" in c.upper()]
    print(f"  CPTAC collections ({len(cptac_all)}):")
    for c in cptac_all:
        print(f"    - {c}")
except Exception as e:
    print(f"  FAIL: {e}")

# 2. TCGA GDC Slide Image 统计 (补 TCGA patients)
print()
print("=" * 60)
print("2. TCGA-LUAD/LUSC Slide Image (GDC)")
print("=" * 60)
filters_tcga = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LUAD", "TCGA-LUSC"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
    ]
}
params2 = {"filters": json.dumps(filters_tcga), "size": "1", "format": "JSON"}
url2 = f"{gdc}/files?{urllib.parse.urlencode(params2)}"
try:
    req = urllib.request.Request(url2, headers={"User-Agent": "curl"})
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read())
    total_tcga = d.get("data",{}).get("pagination",{}).get("total",0)
    print(f"  TCGA-LUAD+LUSC Slide Image files: {total_tcga}")
except Exception as e:
    print(f"  FAIL: {e}")

# 3. 直接查 TCIA TCGA collections (TCGA-LUAD, TCGA-LUSC)
print()
print("=" * 60)
print("3. TCIA TCGA Lung collections")
print("=" * 60)
try:
    for col in ["TCGA-LUAD", "TCGA-LUSC", "TCGA_LUAD", "TCGA_LUSC"]:
        try:
            r = requests.get("https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries", 
                           params={"Collection": col, "format": "JSON"}, timeout=30, verify=False)
            series = r.json()
            if series and isinstance(series, list):
                mods = {}
                for s in series:
                    m = s.get("Modality", "?")
                    mods[m] = mods.get(m, 0) + 1
                total_gb = sum(s.get("FileSize",0) for s in series) / 1e6 / 1024
                pats = len(set(s.get("PatientID","") for s in series))
                print(f"  {col}: {len(series)} series, {mods}, {pats} patients, {total_gb:.1f} GB")
        except Exception as e2:
            print(f"  {col}: FAIL ({e2})")
except Exception as e:
    print(f"  fail: {e}")

print()
print("✅ 结论")
print("-" * 60)
print("  CPTAC 病理 WSI: 不在 TCIA, 可能走 CPTAC Portal 专有下载 (需要 manifest)")
print("  GDC TCGA Slide Image: 几百个 (可直接 GDC API 下载, 免授权)")
print("  TCIA TCGA-LUAD/LUSC: 没查到 (可能 collection 名字不同)")
print()
print("  建议: 先用 GDC 补 TCGA, CPTAC 病理需要单独找 manifest")
