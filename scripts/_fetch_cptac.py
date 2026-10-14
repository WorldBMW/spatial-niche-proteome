"""看原始 CPTAC series 数据, 看 Modality 到底是什么, 然后过滤 + 启动下载"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import requests, pandas as pd
from pathlib import Path

OUT = Path(r"e:\ai_paper\experiment2_fm_time\data\raw\CPTAC")
TCIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
import json

def api(path, **p):
    return requests.get(f"{TCIA}/{path}", params=p, timeout=120, verify=False).json()

print("=== 原始 CPTAC-LUAD series 样本 ===")
luad = api("getSeries", Collection="CPTAC-LUAD", format="JSON")
print(f"Total: {len(luad)}")
# 看前 3 个的所有字段
for s in luad[:3]:
    print(json.dumps(s, indent=2))

print()
print("=== Modality 分布 ===")
modalities = {}
for s in luad:
    m = s.get("Modality", "?")
    modalities[m] = modalities.get(m, 0) + 1
print(f"CPTAC-LUAD: {modalities}")

print()
luad_size_mb = sum(s.get("FileSize", 0) for s in luad) / 1e6
print(f"CPTAC-LUAD 总大小: {luad_size_mb/1024:.1f} GB")
print(f"CPTAC-LUAD patients: {len(set(s.get('PatientID','') for s in luad))}")

print()
print("=== 过滤病理切片 (SM, OTH, 或含 Pathology/Lung 的) ===")
def is_pathology(s):
    mod = s.get("Modality", "")
    if mod in ("SM", "SMO", "OTH"):
        return True
    # ImageType 可能含 Pathology
    img_type = str(s.get("ImageType", "")).lower()
    if "pathology" in img_type:
        return True
    return False

path_only = [s for s in luad if is_pathology(s)]
print(f"病理切片: {len(path_only)} / {len(luad)}")
if path_only:
    print(f"  Patients: {len(set(s['PatientID'] for s in path_only))}")
    total_mb = sum(s.get("FileSize",0) for s in path_only) / 1e6
    print(f"  大小: {total_mb/1024:.1f} GB")

# 同时查 CPTAC-LSCC
print()
print("=== CPTAC-LSCC ===")
lusc = api("getSeries", Collection="CPTAC-LSCC", format="JSON")
print(f"Total: {len(lusc)}")
modalities2 = {}
for s in lusc:
    m = s.get("Modality", "?")
    modalities2[m] = modalities2.get(m, 0) + 1
print(f"Modality: {modalities2}")
path_lusc = [s for s in lusc if is_pathology(s)]
print(f"病理切片: {len(path_lusc)} / {len(lusc)}")
print(f"  Patients: {len(set(s['PatientID'] for s in path_lusc))}")
total_mb_lusc = sum(s.get("FileSize",0) for s in path_lusc) / 1e6
print(f"  大小: {total_mb_lusc/1024:.1f} GB")

# 保存 manifest
all_path = []
for s in path_only + path_lusc:
    all_path.append({
        "SeriesInstanceUID": s.get("SeriesInstanceUID"),
        "StudyInstanceUID": s.get("StudyInstanceUID"),
        "PatientID": s.get("PatientID"),
        "Modality": s.get("Modality"),
        "BodyPart": s.get("BodyPartExamined", ""),
        "NumImages": s.get("NumberofImages", 1),
        "FileSize_MB": round(s.get("FileSize", 0) / 1e6, 1),
        "Collection": s.get("CollectionName", s.get("Collection", "")),
    })

if all_path:
    df = pd.DataFrame(all_path)
    mp = OUT / "cptac_tcia_manifest.csv"
    df.to_csv(mp, index=False)
    print(f"\n✓ manifest 保存: {mp} ({df.shape})")
    print(f"  总计: {df['PatientID'].nunique()} patients, {df['FileSize_MB'].sum()/1024:.1f} GB")
else:
    print("\n❌ 没找到病理切片. CPTAC 在 TCIA 上可能不用标准 Modality 标记.")
    # 存全部 series 让用户看
    all_df = pd.DataFrame([{k:s.get(k) for k in ["SeriesInstanceUID","PatientID","Modality","BodyPartExamined","FileSize"]} for s in luad + lusc])
    mp = OUT / "cptac_tcia_all_series.csv"
    all_df.to_csv(mp, index=False)
    print(f"  已存全部 series → {mp} 供人工查看")
