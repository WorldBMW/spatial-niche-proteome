"""
download_cptac_wsi.py
=====================
下载 CPTAC-LUAD / CPTAC-LUSC 的 WSI .svs 文件。
复用 experiment1 已下载的 gdc-client 和 aspera 传输。

CPTAC NSCLC 队列约 44 例 WSI（去掉 duplicates），共 ~30GB。
"""
import sys, os, subprocess, argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw" / "CPTAC" / "wsi"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EXP1 = ROOT.parent / "experiment1"
GDC_CLIENT = EXP1 / "archive_plan1_canvas" / "gdc-client" / "gdc-client.exe"


def make_manifest() -> Path:
    """构建 CPTAC-NSCLC WSI manifest（从 GDC Data Portal file UUID 列表）。"""
    # 这里先放占位：实际 UUID 从 https://portal.gdc.cancer.gov/ 搜 CPTAC-NSCLC → Image → Diagnostic Slides
    # 示例 manifest 格式:
    manifest = """[
  {"id": "PLACEHOLDER_UUID_1", "filename": "CPTAC-LUAD_001.svs", "data_type": "Slide Image", "experimental_strategy": "Diagnostic Slide"},
  {"id": "PLACEHOLDER_UUID_2", "filename": "CPTAC-LUSC_002.svs", "data_type": "Slide Image", "experimental_strategy": "Diagnostic Slide"}
]"""
    manifest_path = DATA_DIR.parent / "cptac_wsi_manifest.json"
    manifest_path.write_text(manifest)
    return manifest_path


def run_gdc(manifest: Path):
    if not GDC_CLIENT.exists():
        print(f"[ERROR] gdc-client not found: {GDC_CLIENT}")
        print(f"  请先从 https://gdc.cancer.gov/access-data/gdc-data-transfer-tool 下载")
        print(f"  或复用 experiment1/archive_plan1_canvas/gdc-client/gdc-client.exe")
        return

    cmd = [str(GDC_CLIENT), "download", "-m", str(manifest), "-d", str(DATA_DIR), "--no-seg"]
    print(f"[RUN] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[-1000:])


def run_aspera(manifest: Path):
    """备选：aspera 加速下载（experiment1 已安装）。"""
    ascp = Path(r"C:\Program Files (x86)\Common Files\Aspera\bin\ascp.exe")
    if not ascp.exists():
        print(f"[INFO] ascp not found, falling back to gdc-client HTTP")
        run_gdc(manifest)
        return
    # ascp 需要 CPTAC SRA accession IDs，先 run_gdc 拿 UUID 再转


def count_local():
    svs = list(DATA_DIR.rglob("*.svs"))
    total_gb = sum(p.stat().st_size for p in svs) / 1e9
    print(f"[LOCAL] {len(svs)} .svs files, {total_gb:.1f} GB in {DATA_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--aspera", action="store_true")
    args = ap.parse_args()

    if args.count_only:
        count_local()
        sys.exit(0)

    manifest = args.manifest or make_manifest()
    run_fn = run_aspera if args.aspera else run_gdc
    run_fn(manifest)
    count_local()
