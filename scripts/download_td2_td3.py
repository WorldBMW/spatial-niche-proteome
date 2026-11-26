# -*- coding: utf-8 -*-
"""download_td2_td3.py — 下载 TD2/TD3 缺失的 matrix/features 文件."""
import urllib.request, os, gzip, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "visium_luad"

TARGETS = [
    ("TD2", "GSM5702474", [
        ("GSM5702474_TD2_matrix.mtx.gz", "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM5702474&format=file&file=GSM5702474_TD2_matrix.mtx.gz"),
        ("GSM5702474_TD2_features.tsv.gz", "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM5702474&format=file&file=GSM5702474_TD2_features.tsv.gz"),
    ]),
    ("TD3", "GSM5702475", [
        ("GSM5702475_TD3_matrix.mtx.gz", "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM5702475&format=file&file=GSM5702475_TD3_matrix.mtx.gz"),
        ("GSM5702475_TD3_features.tsv.gz", "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM5702475&format=file&file=GSM5702475_TD3_features.tsv.gz"),
    ]),
]

FTP_TEMPLATE = "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{}nnn/GSM{}/suppl/{}"


def download_https(url, out, timeout=600):
    """GEO HTTPS 下载 (可能被 reCAPTCHA 拦截)."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with open(out, "wb") as f:
        f.write(data)
    return len(data)


def download_ftp(gsm, fname, out, timeout=600):
    """NCBI FTP 下载."""
    url = FTP_TEMPLATE.format(gsm[5:8], gsm, fname)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with open(out, "wb") as f:
        f.write(data)
    return len(data)


def verify_gz(path):
    """验证 gzip 文件完整性."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(65536):
                pass
        return True
    except Exception as e:
        return False


def main():
    for slide, gsm, files in TARGETS:
        slide_dir = ROOT / slide
        slide_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\n{slide} ({gsm})\n{'='*60}")
        for fname, https_url in files:
            out = slide_dir / fname
            if out.exists() and verify_gz(out) and out.stat().st_size > 100000:
                print(f"  {fname}: 已存在且完整 ({out.stat().st_size} bytes), 跳过")
                continue
            print(f"  下载 {fname} ...")
            # 尝试 HTTPS
            try:
                size = download_https(https_url, out)
                if verify_gz(out) and size > 100000:
                    print(f"    HTTPS 成功: {size} bytes")
                    continue
                else:
                    print(f"    HTTPS 文件不完整 ({size} bytes), 尝试 FTP")
                    out.unlink(missing_ok=True)
            except Exception as e:
                print(f"    HTTPS 失败: {e}")
            # 尝试 FTP
            try:
                size = download_ftp(gsm, fname, out)
                if verify_gz(out) and size > 100000:
                    print(f"    FTP 成功: {size} bytes")
                else:
                    print(f"    FTP 文件不完整 ({size} bytes)")
                    out.unlink(missing_ok=True)
            except Exception as e:
                print(f"    FTP 失败: {e}")
    print("\n完成")


if __name__ == "__main__":
    main()
