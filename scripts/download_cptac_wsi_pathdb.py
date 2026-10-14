"""
download_cptac_wsi_pathdb.py — 从 TCIA PathDB 选择性下载 CPTAC-LUAD/LSCC 病理 H&E WSI。

来源 (已核实):
  直链模板 = https://pathdb.cancerimagingarchive.net/system/files/wsi/ross/CPTAC/{Tumor}/{Slide_ID}.svs
  节点元数据 = /node/{Pathology}?_format=json  (Tumor/Slide_ID 来自 PathDB 清单)
  免授权, 支持 Range 断点续传; 40x, mpp≈0.4942, 单张约 0.5GB。

清单: data/raw/CPTAC/cptac_wsi_tumor_ffpe_manifest.csv
      (611 张 FFPE 肿瘤片 / 209 例, 由 scripts/_match_wsi_roster.py 生成)

用法 (PowerShell):
  $env:HTTPS_PROXY='http://127.0.0.1:7890'; $env:HTTP_PROXY='http://127.0.0.1:7890'
  python scripts/download_cptac_wsi_pathdb.py --limit 2          # 先测 2 张
  python scripts/download_cptac_wsi_pathdb.py                    # 默认: 每例1张主片 (209 张)
  python scripts/download_cptac_wsi_pathdb.py --all              # 全部 611 张 FFPE 肿瘤片
  python scripts/download_cptac_wsi_pathdb.py --cohort LSCC      # 只下 LUSC
"""
import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "raw" / "CPTAC" / "cptac_wsi_tumor_ffpe_manifest.csv"
OUT_DIR = ROOT / "data" / "raw" / "CPTAC" / "wsi"
STATUS_CSV = ROOT / "data" / "raw" / "CPTAC" / "wsi_download_status.csv"
BASE = "https://pathdb.cancerimagingarchive.net/system/files/wsi/ross/CPTAC"

requests.packages.urllib3.disable_warnings()


def select_slides(per_case: str, cohort: str | None) -> pd.DataFrame:
    df = pd.read_csv(MANIFEST, dtype=str)
    if cohort:
        df = df[df["Tumor"] == cohort.upper()]
    df["pct"] = pd.to_numeric(df["Percent_Tumor_Nuclei"], errors="coerce").fillna(-1)
    df["acc_rank"] = (df["Tumor_Segment_Acceptable"] == "Yes").astype(int)

    if per_case == "one":
        # 每例 1 张主片: 先选 Tumor_Segment_Acceptable=Yes, 再选肿瘤核%最高
        df = df.sort_values(["Case_ID", "acc_rank", "pct", "Slide_ID"],
                            ascending=[True, False, False, True])
        df = df.groupby("Case_ID", as_index=False).head(1)
        print(f"[select] 每例 1 张主片: {len(df)} 张 / {df.Case_ID.nunique()} 例")
    else:
        print(f"[select] 全部 FFPE 肿瘤片: {len(df)} 张 / {df.Case_ID.nunique()} 例")
    return df.sample(frac=1, random_state=2026).reset_index(drop=True)  # 打乱, 两癌种交替


def remote_size(url: str) -> int | None:
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0"},
                         stream=True, timeout=60, verify=False)
        cr = r.headers.get("Content-Range", "")  # bytes 0-0/556308753
        r.close()
        return int(cr.split("/")[-1]) if "/" in cr else None
    except Exception:
        return None


def download_one(row, force: bool) -> str:
    url = f"{BASE}/{row.Tumor}/{row.Slide_ID}.svs"
    out = OUT_DIR / f"{row.Case_ID}__{row.Slide_ID}.svs"
    if not force and out.exists():
        total = remote_size(url)
        if total and abs(out.stat().st_size - total) == 0:
            return "done"

    total = remote_size(url)
    cmd = ["curl.exe", "-L", "-C", "-", "-sS", "--retry", "5",
           "--retry-all-errors", "--retry-delay", "3",
           "--connect-timeout", "30", "-o", str(out), url]
    try:
        subprocess.run(cmd, check=True, timeout=7200)
    except Exception as e:
        print(f"    [ERR] {row.Case_ID}: {e}")
        return "fail"

    if total and out.exists() and out.stat().st_size == total:
        return "done"
    if out.exists() and out.stat().st_size > 1e8:
        return "done_nosize"  # 拿不到总大小但文件已落盘
    return "incomplete"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", dest="per_case", action="store_const", const="all",
                    help="下载全部 611 张 FFPE 肿瘤片 (默认仅每例1张主片)")
    ap.add_argument("--one", dest="per_case", action="store_const", const="one",
                    help="每例 1 张主片 (默认)")
    ap.set_defaults(per_case="one")
    ap.add_argument("--cohort", choices=["LUAD", "LSCC"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="并发下载数 (默认4)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slides = select_slides(args.per_case, args.cohort)
    if args.limit:
        slides = slides.head(args.limit)

    status = {r.Slide_ID: r.status for _, r in pd.read_csv(
        STATUS_CSV).iterrows()} if STATUS_CSV.exists() else {}
    lock = threading.Lock()
    counter = {"ok": 0, "fail": 0, "skip": 0}

    todo = [r for r in slides.itertuples()
            if args.force or status.get(r.Slide_ID) != "done"]
    counter["skip"] = len(slides) - len(todo)
    print(f"[plan] 待下 {len(todo)}, 跳过(已done) {counter['skip']}, 并发 {args.workers}")

    def persist():
        pd.DataFrame(status.items(), columns=["Slide_ID", "status"]).to_csv(
            STATUS_CSV, index=False)

    def worker(idx_total, row):
        i, total = idx_total
        print(f"[{i}/{total}] start {row.Tumor} {row.Case_ID} {row.Slide_ID}", flush=True)
        res = download_one(row, args.force)
        with lock:
            status[row.Slide_ID] = res
            if res.startswith("done"):
                counter["ok"] += 1
                print(f"[{i}/{total}] DONE {row.Case_ID} {row.Slide_ID} "
                      f"(ok={counter['ok']},fail={counter['fail']})", flush=True)
            else:
                counter["fail"] += 1
                print(f"[{i}/{total}] FAIL({res}) {row.Case_ID} {row.Slide_ID}", flush=True)
            persist()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, (i, len(todo)), row)
                for i, row in enumerate(todo, 1)]
        for _ in as_completed(futs):
            pass

    done_n = sum(1 for v in status.values() if v.startswith("done"))
    print(f"\n[DONE] 本次 ok={counter['ok']} fail={counter['fail']} "
          f"skip={counter['skip']} 用时 {(time.time()-t0)/60:.1f} 分")
    print(f"[累计] status=done: {done_n}; 状态表 {STATUS_CSV}")
    n_svs = len(list(OUT_DIR.glob("*.svs")))
    gb = sum(p.stat().st_size for p in OUT_DIR.glob("*.svs")) / 1e9
    print(f"[磁盘] {OUT_DIR.name}/ 下 {n_svs} 个 .svs, {gb:.1f} GB")


if __name__ == "__main__":
    sys.exit(main())
