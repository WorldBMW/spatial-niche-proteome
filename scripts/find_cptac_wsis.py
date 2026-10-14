"""
查询 CPTAC-NSCLC WSI 在 GDC Data Portal 上的 file UUIDs。
然后用 experiment1 里已有的 gdc-client.exe 下载。

用法: python scripts/find_cptac_wsis.py
"""
import requests, json, sys
import pandas as pd
from pathlib import Path

GDC_GRAPHQL = "https://api.gdc.cancer.gov/v0/graphql"
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "CPTAC"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# CPTAC-NSCLC 两个项目
PROJECTS = ["CPTAC-LUAD", "CPTAC-LUSC", "CPTAC-LUAD_LUSC"]


def query_project_slides(project_code: str) -> list:
    """GraphQL 查询 CPTAC 项目下的 WSI .svs 文件。"""
    # 用 GDC REST endpoint 更简洁
    url = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_code]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
            {"op": "in", "content": {"field": "file_format", "value": ["SVS"]}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.project.project_id,file_size,md5sum",
        "format": "JSON",
        "size": 500,
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print(f"  {project_code}: status={r.status_code}")
        return []
    data = r.json()
    files = data.get("data", {}).get("hits", [])
    return files


def main():
    all_files = []
    for proj in PROJECTS:
        print(f"Querying {proj}...")
        files = query_project_slides(proj)
        print(f"  found {len(files)} SVS files")
        for f in files:
            case_ids = [c.get("submitter_id", "") for c in f.get("cases", [])]
            projects = [c.get("project", {}).get("project_id", "") for c in f.get("cases", [])]
            all_files.append({
                "file_id": f["file_id"],
                "file_name": f["file_name"],
                "file_size_mb": f.get("file_size", 0) / 1e6,
                "md5sum": f.get("md5sum", ""),
                "case_id": case_ids[0] if case_ids else "",
                "project": projects[0] if projects else proj,
            })

    if not all_files:
        print("\n[WARN] 没有找到 CPTAC SVS 文件！")
        print("  可能原因: CPTAC-NSCLC 的 WSI 还没在 GDC 公开，或需要换 accession ID")
        print("  手动检查: https://portal.gdc.cancer.gov/repository")
        return

    df = pd.DataFrame(all_files)
    # 去重（同一 file_id 可能多次出现）
    df = df.drop_duplicates(subset=["file_id"])
    out = DATA_DIR / "cptac_wsi_uuids.csv"
    df.to_csv(out, index=False)

    total_gb = df["file_size_mb"].sum() / 1024
    print(f"\n[DONE] {len(df)} unique SVS files → {out}")
    print(f"  预计大小: {total_gb:.1f} GB")
    print(df[["file_id", "file_name", "file_size_mb", "case_id", "project"]].head(10).to_string(index=False))

    # 生成 gdc-client manifest
    manifest = df[["file_id"]].rename(columns={"file_id": "id"})
    manifest_path = DATA_DIR / "gdc_manifest.txt"
    manifest.to_csv(manifest_path, index=False, sep="\t")
    print(f"\n  manifest → {manifest_path}")


if __name__ == "__main__":
    main()
