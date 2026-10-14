"""
download_tcga_wsi.py — 从 GDC Portal 下载 TCGA-LUAD/LUSC H&E WSI
用法: python scripts/download_tcga_wsi.py [--limit N] [--out DIR]
依赖: pip install gdc_client
      或: 从 https://gdc.cancer.gov/access-data/gdc-data-transfer-tool 下载 gdc-client
"""
import os, sys, csv, argparse, subprocess, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_RAW

def fetch_manifest(limit=None):
    """用 GDC API POST query 拿 WSI 文件 manifest (不需要 token, 公开)"""
    import requests
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",
                      "value": ["TCGA-LUAD", "TCGA-LUSC"]}},
            {"op": "in", "content": {"field": "data_type",
                      "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "data_format",
                      "value": ["SVS"]}},
            {"op": "in", "content": {"field": "cases.diagnoses.primary_diagnosis",
                      "value": ["Lung adenocarcinoma",
                                "Lung squamous cell carcinoma"]}},
        ]
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.project.project_id,"
                  "cases.samples.sample_type,cases.samples.sample_id",
        "format": "JSON",
        "size": "2000",
    }
    print("[GDC] 查询公开 manifest...")
    r = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=120)
    r.raise_for_status()
    data = r.json()["data"]["hits"]
    print(f"[GDC] 命中 {len(data)} 个文件")
    
    rows = []
    for f in data:
        proj = f["cases"][0]["project"]["project_id"]
        sample_type = f["cases"][0]["samples"][0]["sample_type"] if f["cases"][0]["samples"] else None
        # 只保留 Primary Tumor / Solid Tissue Normal (sample_type 1 / 11)
        if sample_type and sample_type in ["Primary Tumor", "Solid Tissue Normal"]:
            rows.append({
                "file_id": f["file_id"],
                "file_name": f["file_name"],
                "case_id": f["cases"][0]["submitter_id"],
                "project": proj,
                "sample_type": sample_type,
            })
    
    if limit:
        rows = rows[:limit]
    print(f"[GDC] 筛选后 {len(rows)} 例 WSI (Primary+Normal)")
    return rows

def download_curl(file_id, out_dir):
    """用 curl 直接下载 (GDC 支持, 无需 gdc-client)"""
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    out_file = out_dir / file_id
    if out_file.exists() and out_file.stat().st_size > 1e6:
        print(f"  skip (exists): {file_id}")
        return True
    # curl 支持断点续传 + 进度条
    cmd = ["curl", "-L", "-C", "-", "-o", str(out_file), url,
           "--retry", "3", "--retry-delay", "2",
           "-sS"]
    try:
        subprocess.run(cmd, check=True, timeout=1800)
        return True
    except Exception as e:
        print(f"  FAIL {file_id}: {e}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="下载上限 (测试用, 默认全部)")
    ap.add_argument("--out", type=str, default=str(DATA_RAW / "TCGA" / "wsi"),
                    help="输出目录")
    ap.add_argument("--sample", choices=["primary", "normal", "all"], default="all")
    args = ap.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")
    
    manifest = fetch_manifest(args.limit)
    
    # 写 manifest.csv 方便追踪
    mf_csv = out_dir / "manifest.csv"
    with open(mf_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"[manifest] 已写 {mf_csv}")
    
    # 下载
    ok, fail = 0, 0
    for i, row in enumerate(manifest, 1):
        print(f"[{i}/{len(manifest)}] {row['case_id']} {row['sample_type']} → {row['file_id'][:8]}...")
        if download_curl(row["file_id"], out_dir):
            ok += 1
        else:
            fail += 1
        time.sleep(0.3)  # 礼貌
    
    print(f"\n[DONE] ok={ok} fail={fail}")
    # 统计已下载
    svs = list(out_dir.glob("*.svs"))
    print(f"[out] {len(svs)} .svs in {out_dir}")

if __name__ == "__main__":
    main()
