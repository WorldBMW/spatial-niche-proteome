# -*- coding: utf-8 -*-
"""download_td2_td3_ftp.py — 用 ftplib 直接下载 TD2/TD3 缺失文件."""
import ftplib, gzip, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "visium_luad"

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


def download_ftp(gsm, fname, out, timeout=600):
    ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov", timeout=timeout)
    ftp.login()
    ftp_path = f"/geo/samples/GSM{gsm[3:6]}nnn/{gsm}/suppl/{fname}"
    print(f"  FTP RETR {ftp_path}")
    with open(out, "wb") as f:
        ftp.retrbinary(f"RETR {ftp_path}", f.write)
    ftp.quit()
    return out.stat().st_size


def main():
    for slide, gsm, fname in FILES:
        slide_dir = ROOT / slide
        out = slide_dir / fname
        if out.exists() and verify_gz(out) and out.stat().st_size > 100000:
            print(f"{slide}/{fname}: 已完整 ({out.stat().st_size} bytes), 跳过")
            continue
        print(f"下载 {slide}/{fname} ...")
        try:
            size = download_ftp(gsm, fname, out)
            if verify_gz(out) and size > 100000:
                print(f"  成功: {size} bytes")
            else:
                print(f"  文件不完整 ({size} bytes), 删除")
                out.unlink(missing_ok=True)
        except Exception as e:
            print(f"  失败: {e}")
            out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
