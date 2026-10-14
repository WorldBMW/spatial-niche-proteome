"""
fetch_tcga_os_final.py v5 — 正确处理 namespace BCR XML
"""
import requests, pandas as pd, xml.etree.ElementTree as ET, time, re
from pathlib import Path

RAW_DIR = Path(r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA")
OUT = RAW_DIR / "tcga_os_survival.csv"
XML_DIR = RAW_DIR / "clinical_xmls"
XML_DIR.mkdir(parents=True, exist_ok=True)

# 只解析 (不重新下载, XML 已在本地)
hits = list(XML_DIR.glob("*.xml"))
print(f"[1/3] Using cached XMLs: {len(hits)}")

def strip_ns(tag):
    """去掉 namespace: {http://tcga.nci/...}vital_status → vital_status"""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def find_by_localname(root, localname):
    """递归查找任意 namespace 的 localname 标签"""
    for elem in root.iter():
        if strip_ns(elem.tag) == localname:
            text = elem.text
            if text is not None and text.strip():
                return text.strip()
    return None

def parse_bcr_xml(xml_bytes, fname):
    try:
        root = ET.fromstring(xml_bytes)
    except:
        return None
    
    m = re.search(r"(TCGA-\d{2}-\d{4})", fname)
    case_id = m.group(1) if m else None
    
    vital = find_by_localname(root, "vital_status")
    days_death = find_by_localname(root, "days_to_death")
    days_lfu = find_by_localname(root, "days_to_last_followup")
    
    # OS months: 优先用 days_to_death (DECEASED), 其次 days_to_last_followup (ALIVE)
    os_months = None
    os_status = vital  # Alive / Dead
    
    if days_death and days_death != "[Not Available]" and days_death != "[Not Applicable]":
        try:
            os_months = round(float(days_death) / 30.44, 1)
            os_status = "DECEASED"
        except: pass
    elif days_lfu and days_lfu != "[Not Available]" and days_lfu != "[Not Applicable]":
        try:
            os_months = round(float(days_lfu) / 30.44, 1)
            if os_status is None: os_status = "ALIVE"
        except: pass
    elif vital:
        # 有 vital status 但没有 days → 标记但缺失时间
        if vital.lower() == "dead": os_status = "DECEASED"
        elif vital.lower() == "alive": os_status = "ALIVE"
    
    # DFS (如果有的话)
    days_dfs = find_by_localname(root, "days_to_dfs") or find_by_localname(root, "days_to_disease_free")
    dfs_months = None
    if days_dfs and days_dfs != "[Not Available]" and days_dfs != "[Not Applicable]":
        try: dfs_months = round(float(days_dfs)/30.44, 1)
        except: pass
    
    # AJCC stage
    stage = (find_by_localname(root, "ajcc_pathologic_stage") or 
             find_by_localname(root, "pathologic_stage") or
             find_by_localname(root, "stage"))
    
    return {
        "case_id": case_id, "file_name": fname,
        "OS_MONTHS": os_months, "OS_STATUS": os_status,
        "vital_status_raw": vital,
        "days_to_death": days_death, "days_to_last_followup": days_lfu,
        "DFS_MONTHS": dfs_months,
        "AJCC_STAGE": stage,
    }

results = []
for i, xml_path in enumerate(hits):
    try:
        row = parse_bcr_xml(xml_path.read_bytes(), xml_path.name)
        if row:
            # 从文件名判断 LUAD/LUSC
            if "luad" in xml_path.name.lower() or "LUAD" in xml_path.name:
                row["project"] = "TCGA-LUAD"
            elif "lusc" in xml_path.name.lower() or "LUSC" in xml_path.name:
                row["project"] = "TCGA-LUSC"
            results.append(row)
    except Exception as e:
        pass
    
    if (i+1) % 200 == 0:
        print(f"  [{i+1}/{len(hits)}] parsed")

print(f"\n[DONE] parsed: {len(results)}")

df = pd.DataFrame(results)
# 去重 (同一个 case_id 可能多个 XML)
df = df.drop_duplicates(subset="case_id", keep="first")
print(f"\n=== TCGA OS Survival ===")
print(f"Total unique patients: {len(df)}")
print(f"  OS_MONTHS non-null: {df['OS_MONTHS'].notna().sum()}")
print(f"  OS_STATUS non-null: {df['OS_STATUS'].notna().sum()}")
print(f"  OS_STATUS distribution:")
print(df["OS_STATUS"].value_counts())
print(f"  OS_MONTHS stats: mean={df['OS_MONTHS'].mean():.1f}, median={df['OS_MONTHS'].median():.1f}")
print(f"  AJCC_STAGE non-null: {df['AJCC_STAGE'].notna().sum()}")

# project 分布
if "project" in df.columns:
    print(f"\n  project: {df['project'].value_counts().to_dict()}")

df.to_csv(OUT, index=False)
print(f"\nSaved → {OUT}")
print(f"\n=== Sample (10 rows with OS) ===")
with_os = df[df["OS_MONTHS"].notna()].head(10)
print(with_os[["case_id", "OS_MONTHS", "OS_STATUS", "AJCC_STAGE"]].to_string())
