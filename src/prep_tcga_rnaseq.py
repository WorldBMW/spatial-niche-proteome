"""P1: 从 toil pan-cancer RSEM 矩阵提取各癌种 RNA-seq CSV。

输入:
  data/raw/TCGA/rnaseq/tcga_RSEM_gene_fpkm.gz   (toil-xena-hub, log2(fpkm+0.001), 行=gencode id, 列=16位 barcode)
  data/raw/TCGA/case_lists/tcga_{can}_cases.txt (GDC 官方 case_id)
输出:
  data/raw/TCGA/rnaseq/tcga_{can}_rnaseq.csv    (行=case_id 12位, 列=基因 symbol)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CANON = ["gbm", "hnsc", "kirc", "ucec"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--probe-map", required=True)
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. probeMap: gencode id (带版本) -> gene symbol
    pm = pd.read_csv(args.probe_map, sep="\t")
    id_col = "name" if "name" in pm.columns else pm.columns[3]
    sym_col = "symbol" if "symbol" in pm.columns else pm.columns[-1]
    g2s = dict(zip(pm[id_col].astype(str), pm[sym_col].astype(str)))
    print(f"[probeMap] {len(g2s)} gencode id -> symbol")

    # 2. 官方 case 列表
    case_lists = {}
    for can in CANON:
        fp = Path(args.case_dir) / f"tcga_{can}_cases.txt"
        with open(fp) as f:
            case_lists[can] = set(f.read().split())
        print(f"[cases] {can}: {len(case_lists[can])}")

    # 3. 读 header, 确定各癌种保留列
    header = pd.read_csv(args.matrix, sep="\t", nrows=0).columns.tolist()
    samples = header[1:]                      # 第 0 列是 gencode id
    case_of = [s[:12] for s in samples]
    keep_idx = {}                             # can -> [(col_pos, case_id, sample_barcode)]
    for can, cases in case_lists.items():
        keep_idx[can] = [(i, case_of[i], samples[i]) for i in range(len(samples))
                         if case_of[i] in cases]
        print(f"[header] {can}: {len(keep_idx[can])} 个样本列")

    # 4. 读矩阵 (只读需要的列)
    all_keep = sorted({i for lst in keep_idx.values() for i, _, _ in lst})
    usecols = [header[0]] + [header[i + 1] for i in all_keep]
    print(f"[matrix] 读取 {len(usecols)} 列 ...")
    mat = pd.read_csv(args.matrix, sep="\t", usecols=usecols, index_col=0)
    print(f"[matrix] {mat.shape[0]} genes x {mat.shape[1]} samples")

    # 5. 按癌种输出: 同 case 多样本取 01 原发瘤优先, 否则首个
    for can, cols in keep_idx.items():
        per_case = {}
        for i, cid, bar in cols:
            col_name = header[i + 1]
            per_case.setdefault(cid, []).append((bar.startswith("TCGA-") and bar[13:15] == "01", col_name))
        rows = {}
        for cid, lst in per_case.items():
            lst.sort(key=lambda t: (not t[0],))          # 原发 (01) 优先
            rows[cid] = mat[lst[0][1]]
        df = pd.DataFrame(rows).T                        # index=case_id
        df.index.name = "case_id"
        # gencode id -> symbol; 保留能映射的, 同名取均值
        df.columns = [g2s.get(str(c), None) for c in df.columns]
        df = df.loc[:, df.columns.notna()]
        df = df.T.groupby(level=0).mean().T              # 同 symbol 去重
        out = out_dir / f"tcga_{can}_rnaseq.csv"
        df.to_csv(out)
        print(f"[out] {can}: {df.shape[0]} cases x {df.shape[1]} genes -> {out}")

if __name__ == "__main__":
    main()
