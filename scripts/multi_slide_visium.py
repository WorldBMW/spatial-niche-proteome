# -*- coding: utf-8 -*-
"""
multi_slide_visium.py — 多切片 Visium 空间验证 (TD1/TD2/TD3)

对 GSE189487 的多个 Visium 切片计算 SLPI 的 Moran's I，
把"单切片单基因"升级为 n=3 切片验证。

用法:
  python scripts/multi_slide_visium.py
  python scripts/multi_slide_visium.py --slides TD1 TD2 TD3
"""
import sys, os, argparse, gzip, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.spatial import cKDTree
from math import erf, sqrt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config

DATA_ROOT = config.DATA_RAW.parent / "visium_luad"
GENE = "SLPI"
K_NN = 6
N_PERM = 1000
RNG_SEED = 42


def read_tsv_gz(path, min_cols=1):
    rows = []
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= min_cols:
                rows.append(parts)
    return rows


def read_gene_row_from_mtx(path, gene_idx, n_cols):
    """从 mtx.gz 中只读目标基因行 + 列总和 (一次遍历, 内存友好)."""
    gene_row = 1 + gene_idx  # mtx 是 1-based
    gene_vec = np.zeros(n_cols, dtype=float)
    col_sums = np.zeros(n_cols, dtype=float)
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            if line.startswith("%"):
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
                col_sums[j - 1] += v
                if i == gene_row:
                    gene_vec[j - 1] = v
    return gene_vec, col_sums


def morans_i(values, coords, k=K_NN, n_perm=N_PERM, seed=RNG_SEED):
    """Moran's I: KNN 权重 + 行标准化，解析 z + 置换检验。"""
    n = len(values)
    if n < k + 2:
        return dict(I=np.nan, EI=np.nan, z=np.nan, p_norm=np.nan, p_perm=np.nan, n=n)
    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=k + 1)
    nbr = idx[:, 1:]
    x = values.astype(float)
    x_dev = x - x.mean()
    lag = x_dev[nbr].mean(axis=1)
    denom = (x_dev ** 2).sum()
    if denom == 0:
        return dict(I=np.nan, EI=-1.0/(n-1), z=None, p_norm=None, p_perm=None, n=n)
    I = (x_dev * lag).sum() / denom

    W = np.zeros((n, n))
    for i in range(n):
        W[i, nbr[i]] = 1.0 / k
    S0 = W.sum()
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(axis=0) + W.sum(axis=1)) ** 2).sum()
    b2 = (n * ((x_dev ** 2) ** 2).sum()) / (denom ** 2)
    EI = -1.0 / (n - 1)
    var_num = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2)
    var_den = (n - 1) * (n - 2) * (n - 3) * S0**2
    var_norm = (var_num - b2 * (n*(n-1)*S1 - 2*n*S2 + 6*S0**2)) / var_den
    var_norm = abs(var_norm)
    z = (I - EI) / np.sqrt(var_norm) if var_norm > 0 else np.nan
    p_norm = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))) if z == z else np.nan

    rng = np.random.default_rng(seed)
    perm_I = np.empty(n_perm)
    for b in range(n_perm):
        x_perm = rng.permutation(x)
        xd = x_perm - x_perm.mean()
        lag_p = xd[nbr].mean(axis=1)
        d_p = (xd ** 2).sum()
        perm_I[b] = (xd * lag_p).sum() / d_p if d_p > 0 else np.nan
    p_perm = (np.sum(perm_I >= I) + 1) / (n_perm + 1) if np.any(~np.isnan(perm_I)) else np.nan
    return dict(I=round(float(I), 6), EI=round(float(EI), 6),
                z=round(float(z), 4) if z == z else np.nan,
                p_norm=round(float(p_norm), 6) if p_norm == p_norm else np.nan,
                p_perm=round(float(p_perm), 6) if p_perm == p_perm else np.nan,
                n=int(n))


def analyze_slide(slide_dir, gene=GENE):
    """分析单个 Visium 切片。"""
    slide_dir = Path(slide_dir)
    name = slide_dir.name

    # 找文件
    pattern_map = {
        "barcodes": "*_barcodes.tsv.gz",
        "features": "*_features.tsv.gz",
        "matrix": "*_matrix.mtx.gz",
        "positions": "*_tissue_positions_list.csv.gz",
    }
    files = {}
    for key, pat in pattern_map.items():
        matches = list(slide_dir.glob(pat))
        if not matches:
            print(f"  [{name}] 缺少 {key} 文件")
            return None
        files[key] = matches[0]

    # 读 features → 找 gene 行号
    feat_rows = read_tsv_gz(files["features"])
    gene_idx = None
    for i, row in enumerate(feat_rows):
        if row[0] == gene or (len(row) > 1 and row[1] == gene):
            gene_idx = i
            break
    if gene_idx is None:
        print(f"  [{name}] {gene} 不在 features 中")
        return None

    # 读 barcodes
    barcodes = [r[0] for r in read_tsv_gz(files["barcodes"])]

    # 读 matrix (只取目标基因行 + 列总和, 内存友好)
    result = read_gene_row_from_mtx(files["matrix"], gene_idx, len(barcodes))
    if result is None:
        print(f"  [{name}] 无法读取 {gene} 行")
        return None
    gene_vec, col_sums = result

    # 读 tissue positions (CSV, 逗号分隔)
    pos_rows = []
    with gzip.open(files["positions"], "rt", errors="ignore") as fh:
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= 4:
                pos_rows.append(p)
    pos_dict = {}
    for r in pos_rows:
        bc = r[0]
        try:
            in_tissue = int(r[1])
        except (ValueError, IndexError):
            in_tissue = 1
        if in_tissue == 0:
            continue
        try:
            pxl_col = int(r[5]); pxl_row = int(r[4])  # pxl_col, pxl_row (像素坐标, 与 slpi_spatial_analysis.py 一致)
            pos_dict[bc] = (pxl_col, pxl_row)
        except (ValueError, IndexError):
            continue

    # 对齐
    common_bcs = [bc for bc in barcodes if bc in pos_dict]
    if len(common_bcs) < 50:
        print(f"  [{name}] 有效 spots 不足: {len(common_bcs)}")
        return None

    bc_to_idx = {bc: i for i, bc in enumerate(barcodes)}
    expr = np.array([gene_vec[bc_to_idx[bc]] for bc in common_bcs], dtype=float)
    coords = np.array([pos_dict[bc] for bc in common_bcs], dtype=float)

    # CPM 归一化 (1e4, 与 slpi_spatial_analysis.py 一致)
    cpm = np.array([
        expr[i] / max(col_sums[bc_to_idx[common_bcs[i]]], 1) * 1e4
        for i in range(len(common_bcs))
    ])
    log_cpm = np.log1p(cpm)

    # Moran's I
    result = morans_i(log_cpm, coords)
    result["slide"] = name
    result["gene"] = gene
    result["n_spots"] = len(common_bcs)
    result["mean_expr"] = round(float(np.mean(log_cpm)), 4)
    result["pct_expressing"] = round(float(np.mean(expr > 0) * 100), 2)
    z_val = result["z"]
    p_val = result["p_perm"]
    z_str = f"{z_val:.3f}" if z_val is not None and z_val == z_val else "N/A"
    p_str = f"{p_val:.4f}" if p_val is not None and p_val == p_val else "N/A"
    print(f"  [{name}] {gene} Moran's I={result['I']:.4f}, z={z_str}, "
          f"p_perm={p_str}, n={result['n_spots']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", default=["TD1_IAC", "TD2", "TD3"],
                    help="Visium 切片目录名")
    ap.add_argument("--gene", nargs="+", default=["SLPI"],
                    help="要验证的基因 (可多个)")
    ap.add_argument("--out", default=str(config.ROOT / "results_dict_encoder_sensitivity" / "visium_multi_slide_summary.csv"))
    args = ap.parse_args()

    print(f"{'='*60}\n多切片 Visium 空间验证: {args.gene}\n{'='*60}")
    results = []
    for slide in args.slides:
        slide_dir = DATA_ROOT / slide
        if not slide_dir.exists():
            print(f"  [SKIP] {slide_dir} 不存在")
            continue
        print(f"\n--- {slide} ---")
        for gene in args.gene:
            r = analyze_slide(slide_dir, gene=gene)
            if r:
                results.append(r)

    if not results:
        print("无有效结果")
        return

    df = pd.DataFrame(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n{'='*60}\n汇总 ({len(results)} 切片)\n{'='*60}")
    print(df[["slide", "gene", "n_spots", "I", "z", "p_norm", "p_perm", "pct_expressing"]].to_string(index=False))

    # 统计: 有多少切片显著
    n_sig = int((df["p_perm"] < 0.05).sum())
    print(f"\n显著 (p_perm<0.05): {n_sig}/{len(results)} 切片")
    print(f"结果保存到: {out}")


if __name__ == "__main__":
    main()
