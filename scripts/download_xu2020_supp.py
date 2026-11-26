# -*- coding: utf-8 -*-
"""下载 Xu et al. 2020 Cell 补充表 mmc1 (临床) 与 mmc4 (蛋白组 103 例)."""
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "external" / "xu2020_luad"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://ars.els-cdn.com/content/image"
PII = "S0092867420306760"  # Xu et al. Cell 2020, 10.1016/j.cell.2020.05.043
FILES = {
    "mmc1.xlsx": f"{BASE}/1-s2.0-{PII}-mmc1.xlsx",  # Table S1 临床/病理
    "mmc4.xlsx": f"{BASE}/1-s2.0-{PII}-mmc4.xlsx",  # Table S4 蛋白组 103 例
}

H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cell.com/cell/fulltext/S0092-8674(20)30676-0",
}

for name, url in FILES.items():
    dst = OUT / name
    if dst.exists() and dst.stat().st_size > 10000:
        print(f"{name}: 已存在 ({dst.stat().st_size} bytes)")
        continue
    try:
        req = urllib.request.Request(url, headers=H)
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        dst.write_bytes(data)
        print(f"{name}: OK {len(data)} bytes -> {dst}")
    except Exception as e:
        print(f"{name}: ERR {str(e)[:150]}")
