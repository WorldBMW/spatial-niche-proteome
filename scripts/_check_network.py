"""本地下载前置检查: TCIA 连通性 + manifest 规模 + 磁盘空间 + 网络代理"""
import pandas as pd, json, os, requests

print("=" * 60)
print("1. TCIA API 连通性测试")
print("=" * 60)
TCIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
try:
    r = requests.get(f"{TCIA}/getCollectionValues", timeout=15, verify=False)
    print(f"  TCIA API: {r.status_code}")
    if r.status_code == 200:
        cols = [c for c in r.json() if isinstance(c,str) and "CPTAC" in c and "LUNG" in c.upper()]
        print(f"  CPTAC-Lung collections: {cols}")
except Exception as e:
    print(f"  TCIA FAIL: {e}")
    print("  尝试跳过 SSL 验证...")
    import urllib3; urllib3.disable_warnings()
    try:
        r = requests.get(f"{TCIA}/getCollectionValues", timeout=15, verify=False)
        print(f"  TCIA (no verify): {r.status_code}")
    except Exception as e2:
        print(f"  还是 FAIL: {e2}")

print()
print("=" * 60)
print("2. Manifest 规模")
print("=" * 60)
mp = r"e:\ai_paper\experiment2_fm_time\data\raw\CPTAC\cptac_tcia_manifest.csv"
if os.path.exists(mp):
    df = pd.read_csv(mp)
    print(f"  manifest: {df.shape}")
    print(f"  cols: {list(df.columns)}")
    if "Modality" in df.columns:
        print(f"  Modality: {df['Modality'].value_counts().to_dict()}")
    if "FileSize_MB" in df.columns:
        total_gb = df["FileSize_MB"].sum() / 1024
        print(f"  估算总大小: {total_gb:.1f} GB")
    print(f"  Collections: {df['Collection'].value_counts().to_dict()}")
    if "PatientID" in df.columns:
        print(f"  唯一 patients: {df['PatientID'].nunique()}")
else:
    print(f"  manifest 不存在: {mp}")

print()
print("=" * 60)
print("3. 磁盘空间")
print("=" * 60)
try:
    import shutil
    total, used, free = shutil.disk_usage(r"e:\ai_paper")
    print(f"  E: {total/1024**3:.1f} GB total, {used/1024**3:.1f} GB used, {free/1024**3:.1f} GB free")
except Exception as e:
    print(f"  FAIL: {e}")

print()
print("=" * 60)
print("4. 已有本地 .svs / .zip")
print("=" * 60)
import glob
for label, pattern in [
    ("TCGA .svs", r"e:\ai_paper\**\*.svs"),
    ("CPTAC 已下 .zip", r"e:\ai_paper\experiment2_fm_time\data\raw\CPTAC\wsi\*.zip"),
]:
    files = glob.glob(pattern, recursive=True)
    total_mb = sum(os.path.getsize(f) for f in files) / 1e6 if files else 0
    print(f"  {label}: {len(files)} files, {total_mb:.0f} MB")

print()
print("=" * 60)
print("5. 网络代理解析")
print("=" * 60)
for env in ["HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"]:
    v = os.environ.get(env, "")
    if v:
        print(f"  {env}={v}")
if not any(os.environ.get(e) for e in ["HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"]):
    print("  无代理设置")
print(f"  代理可能需要手动设置 (公司内网常用)")
