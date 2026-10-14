"""
从 TCIA (Cancer Imaging Archive) 下载 CPTAC-LUAD / CPTAC-LSCC WSI。
TCIA API 免授权：https://services.cancerimagingarchive.net/

用法: python scripts/download_cptac_from_tcia.py [--collection CPTAC-LUAD] [--limit 10]
"""
import sys, os, time, argparse
from pathlib import Path
import requests
import pandas as pd

OUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "CPTAC" / "wsi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def query_collection(collection: str) -> pd.DataFrame:
    """查询 TCIA collection 下的 Series (一组同病例的切片)。"""
    print(f"[TCIA] 查询 collection={collection} ...")
    r = requests.get(
        f"{TCIA_BASE}/getSeries",
        params={"Collection": collection, "format": "JSON"},
        timeout=60,
    )
    if r.status_code != 200:
        print(f"  FAIL: status={r.status_code} {r.text[:200]}")
        return pd.DataFrame()
    series = r.json()
    print(f"  {collection}: {len(series)} series")

    rows = []
    for s in series:
        rows.append({
            "SeriesInstanceUID": s.get("SeriesInstanceUID"),
            "StudyInstanceUID": s.get("StudyInstanceUID"),
            "PatientID": s.get("PatientID"),
            "Modality": s.get("Modality"),
            "BodyPart": s.get("BodyPartExamined"),
            "NumImages": s.get("NumberofImages", 1),
            "FileSize_MB": round(s.get("FileSize", 0) / 1e6, 1),
            "Collection": collection,
        })
    df = pd.DataFrame(rows)
    return df


def download_series(series_uid: str, out_dir: Path) -> bool:
    """下载一个 Series 的所有文件。"""
    url = f"{TCIA_BASE}/getImage?SeriesInstanceUID={series_uid}"
    out_file = out_dir / f"{series_uid}.zip"
    if out_file.exists() and out_file.stat().st_size > 1e6:
        return True  # skip
    try:
        r = requests.get(url, stream=True, timeout=300, verify=False)
        if r.status_code != 200:
            print(f"  [FAIL] {series_uid}: {r.status_code}")
            return False
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(out_file, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                downloaded += len(chunk)
        if total > 0:
            print(f"  saved {out_file.name} ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [ERR] {series_uid}: {e}")
        if out_file.exists():
            out_file.unlink()
        return False


def collections_available():
    """列出 CPTAC 相关 collection。"""
    r = requests.get(f"{TCIA_BASE}/getCollectionValues", timeout=30)
    raw = r.json()
    # API 可能返回 list[str] 或 list[dict]
    cols = []
    for item in raw:
        if isinstance(item, dict):
            name = item.get("Collection", item.get("name", str(item)))
        else:
            name = str(item)
        if "CPTAC" in name.upper() and any(k in name.upper() for k in ["LUAD", "LSCC", "NSCLC", "LUNG"]):
            cols.append(name)
    print(f"TCIA 中 CPTAC-Lung 相关 collections: {cols}")
    return cols if cols else ["CPTAC-LUAD", "CPTAC-LSCC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", choices=["CPTAC-LUAD", "CPTAC-LSCC", "ALL"], default="ALL")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    # 先列出可用
    cols = collections_available()

    targets = []
    if args.collection == "ALL":
        targets = ["CPTAC-LUAD", "CPTAC-LSCC"]
    else:
        targets = [args.collection]

    all_df = []
    for t in targets:
        df = query_collection(t)
        if not df.empty:
            all_df.append(df)
    if not all_df:
        print("[WARN] 没有可下载的 CPTAC Lung 数据")
        return

    df = pd.concat(all_df, ignore_index=True)
    df = df[df["Modality"] == "SM"].copy()  # pathology slide 通常是 SM
    print(f"\n合计 {len(df)} series 待下载")
    print(df.groupby("Collection")["PatientID"].nunique())

    manifest_path = OUT_DIR.parent / "cptac_tcia_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"manifest → {manifest_path}")

    if args.list_only:
        return

    # 下载
    n = min(len(df), args.limit) if args.limit else len(df)
    ok = 0
    for i, row in df.head(n).iterrows():
        print(f"\n[{i+1}/{n}] {row['Collection']} Patient={row['PatientID']} Series={row['SeriesInstanceUID']}")
        if download_series(row["SeriesInstanceUID"], OUT_DIR):
            ok += 1
        time.sleep(0.5)

    total_mb = sum(p.stat().st_size for p in OUT_DIR.glob("*.zip")) / 1e6
    print(f"\n[DONE] {ok}/{n} series, {total_mb:.0f} MB → {OUT_DIR}")


if __name__ == "__main__":
    main()
