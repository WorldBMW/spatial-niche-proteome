# -*- coding: utf-8 -*-
"""download_td2_td3_v2.py — 多源下载 TD2/TD3 缺失文件."""
import urllib.request, gzip, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "visium_luad"

# 多个下载源
SOURCES = [
    # NCBI HTTPS (FTP 服务器的 HTTPS 接口)
    lambda gsm, fname: f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{gsm[3:7]}nnn/{gsm}/suppl/{fname}",
    # DDBJ 镜像
    lambda gsm, fname: f"https://ftp.ddbj.nig.ac.jp/geo/samples/GSM{gsm[3:7]}nnn/{gsm}/suppl/{fname}",
    # GEO HTTPS download API
    lambda gsm, fname: f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gsm}&format=file&file={fname}",
]

FILES = [
    ("TD2", "GSM5702474", "GSM5702474_TD2_matrix.mtx.gz"),
    ("TD2", "GSM5702474", "GSM5702474_TD2_features.tsv.gz"),
    ("TD3", "GSM5702475", "GSM5702475_TD3_matrix.mtx.gz"),
    ("TD3", "GSM5702475", "GSM5702475_TD3_features.tsv.gz"),
]


def verify_gz(path):
    try:
        with gzip.open(path, "rb") as f:
            while f.read(65536):
                pass
        return True
    except Exception:
        return False


def download_url(url, out, timeout=300):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with open(out, "wb") as f:
        f.write(data)
    return len(data)


def main():
    for slide, gsm, fname in FILES:
        slide_dir = ROOT / slide
        slide_dir.mkdir(parents=True, exist_ok=True)
        out = slide_dir / fname
        if out.exists() and verify_gz(out) and out.stat().st_size > 100000:
            print(f"{slide}/{fname}: 已完整 ({out.stat().st_size} bytes), 跳过")
            continue
        print(f"\n下载 {slide}/{fname} ...")
        success = False
        for i, source_fn in enumerate(SOURCES):
            url = source_fn(gsm, fname)
            src_name = ["NCBI-HTTPS", "DDBJ", "GEO-API"][i]
            try:
                print(f"  尝试 {src_name}: {url[:80]}...")
                t0 = time.time()
                size = download_url(url, out)
                elapsed = time.time() - t0
                if verify_gz(out) and size > 100000:
                    print(f"    成功: {size} bytes in {elapsed:.0f}s")
                    success = True
                    break
                else:
                    print(f"    文件不完整 ({size} bytes)")
                    out.unlink(missing_ok=True)
            except Exception as e:
                print(f"    失败: {str(e)[:100]}")
                out.unlink(missing_ok=True)
        if not success:
            print(f"  所有源均失败")


if __name__ == "__main__":
    main()
