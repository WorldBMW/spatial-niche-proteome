"""
SLPI 空间分布独立验证分析（自包含版，仅依赖 numpy/scipy/pandas/matplotlib）
数据: GEO GSE189487 / GSM5702473 (TD1, IAC, LUAD Visium)

分析内容:
1. 读取 10x Visium 三件套 (features / barcodes / tissue_positions) + matrix
2. 提取 SLPI 每个 spot 的表达量（CPM 归一化 + log1p）
3. 计算 Moran's I（KNN k=6 空间权重，行标准化；解析 z 检验 + 1000 次置换检验）
4. 输出: 结果 txt、spot 级 CSV、空间热图 PNG

运行: C:\\Users\\jm\\anaconda3\\python.exe scripts\\slpi_spatial_analysis.py
"""
import gzip
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.spatial import cKDTree

DATA_DIR = r"f:\ai_paper\experiment2_fm_time\data\visium_luad\TD1_IAC"
OUT_DIR = r"f:\ai_paper\experiment2_fm_time\data\visium_luad"
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


def morans_i(values, coords, k=K_NN, n_perm=N_PERM, seed=RNG_SEED):
    """自实现 Moran's I：KNN 权重 + 行标准化，解析 z 与置换检验。"""
    n = len(values)
    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=k + 1)          # 第 0 列为自身
    nbr = idx[:, 1:]
    # 行标准化权重: 每个 spot 的邻居权重 = 1/k
    x = values.astype(float)
    x_dev = x - x.mean()
    # 拉格朗日形式: I = (n / S0) * (sum_ij w_ij z_i z_j) / (sum_i z_i^2)
    # 行标准化后 S0 = n
    lag = x_dev[nbr].mean(axis=1)
    denom = (x_dev ** 2).sum()
    if denom == 0:
        return dict(I=np.nan, EI=-1.0 / (n - 1), z=None, p_norm=None, p_perm=None)
    I = (n / n) * (x_dev * lag).sum() / denom       # = sum(z_i * lag_i)/sum(z_i^2)

    # 解析方差（正态假设）
    S0 = n
    S1 = n * (1.0 / k + (k - 1) * (1.0 / k) ** 2 * 0)  # 行标准化近似的取法
    # 更稳妥：直接由权重矩阵统计量计算
    W = np.zeros((n, n))
    for i in range(n):
        W[i, nbr[i]] = 1.0 / k
    S0 = W.sum()
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(axis=0) + W.sum(axis=1)) ** 2).sum()
    b2 = (n * ((x_dev ** 2) ** 2).sum()) / (denom ** 2) if denom > 0 else 0
    EI = -1.0 / (n - 1)
    var_num = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    var_den = (n - 1) * (n - 2) * (n - 3) * S0 ** 2
    var_norm = (var_num - b2 * (n * (n - 1) * S1 - 2 * n * S2 + 6 * S0 ** 2)) / var_den
    var_norm = abs(var_norm)
    z = (I - EI) / np.sqrt(var_norm) if var_norm > 0 else np.nan
    from math import erf, sqrt
    p_norm = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))) if z == z else np.nan

    # 置换检验（空间随机化，单侧: 观测值大于随机值）
    rng = np.random.default_rng(seed)
    perm_I = np.empty(n_perm)
    for b in range(n_perm):
        xs = rng.permutation(x)
        zs = xs - xs.mean()
        perm_I[b] = (zs * zs[nbr].mean(axis=1)).sum() / (zs ** 2).sum()
    p_perm = (np.sum(perm_I >= I) + 1) / (n_perm + 1)

    return dict(I=I, EI=EI, z=z, p_norm=p_norm, p_perm=p_perm,
                perm_mean=perm_I.mean(), perm_sd=perm_I.std(ddof=1), n=n, k=k, S0=S0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 1. features ----------
    feat_path = os.path.join(DATA_DIR, "GSM5702473_TD1_features.tsv.gz")
    feats = read_tsv_gz(feat_path, 3)
    genes = [r[1] for r in feats]
    if GENE not in genes:
        sys.exit(f"ERROR: {GENE} 不在 features 中")
    slpi_row = genes.index(GENE)                      # 0-based → matrix 行号（无表头）
    print(f"[1] 基因数={len(genes)}  {GENE} 行号(0-based)={slpi_row}")

    # ---------- 2. barcodes ----------
    bc_path = os.path.join(DATA_DIR, "GSM5702473_TD1_barcodes.tsv.gz")
    barcodes = [r[0] for r in read_tsv_gz(bc_path, 1)]
    print(f"[2] barcodes={len(barcodes)}")

    # ---------- 3. tissue positions ----------
    pos_path = os.path.join(DATA_DIR, "GSM5702473_TD1_tissue_positions_list.csv.gz")
    pos_rows = []
    with gzip.open(pos_path, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= 6:
                pos_rows.append((p[0], int(p[1]), int(p[2]), int(p[3]),
                                 float(p[4]), float(p[5])))
    pos = pd.DataFrame(pos_rows, columns=["barcode", "in_tissue", "array_row",
                                          "array_col", "pxl_row", "pxl_col"])
    print(f"[3] positions={len(pos)}  组织内={int((pos.in_tissue == 1).sum())}")

    # ---------- 4. matrix ----------
    mtx_path = os.path.join(DATA_DIR, "GSM5702473_TD1_matrix.mtx.gz")
    print(f"[4] 读取 matrix ({os.path.getsize(mtx_path)/1e6:.1f} MB 压缩) ...")
    mat = mmread(mtx_path).tocsc()
    print(f"    矩阵形状 (genes x spots) = {mat.shape}")
    assert mat.shape[0] == len(genes), "features 与 matrix 行数不一致"
    assert mat.shape[1] == len(barcodes), "barcodes 与 matrix 列数不一致"

    raw = np.asarray(mat[slpi_row, :].todense()).ravel()

    # ---------- 5. 归一化：每 spot 总计数 CPM(1e4) + log1p ----------
    total = np.asarray(mat.sum(axis=0)).ravel()
    cpm = np.divide(raw * 1e4, total, out=np.zeros_like(raw, dtype=float), where=total > 0)
    log_expr = np.log1p(cpm)

    df = pos.copy()
    df["SLPI_raw"] = df["barcode"].map(dict(zip(barcodes, raw)))
    df["SLPI_log_cpm"] = df["barcode"].map(dict(zip(barcodes, log_expr)))
    df["spot_total_counts"] = df["barcode"].map(dict(zip(barcodes, total)))
    df = df.dropna(subset=["SLPI_raw"]).copy()
    tissue = df[df.in_tissue == 1].copy()

    print(f"[5] 组织内 spots={len(tissue)}  检出SLPI>0={int((tissue.SLPI_raw > 0).sum())}"
          f" ({(tissue.SLPI_raw > 0).mean()*100:.1f}%)")
    print(f"    SLPI raw: mean={tissue.SLPI_raw.mean():.2f} median={tissue.SLPI_raw.median():.1f} "
          f"max={tissue.SLPI_raw.max():.0f}")

    # ---------- 6. Moran's I ----------
    coords = tissue[["pxl_col", "pxl_row"]].to_numpy(dtype=float)  # x=col, y=row
    res_log = morans_i(tissue["SLPI_log_cpm"].to_numpy(), coords)
    res_raw = morans_i(tissue["SLPI_raw"].to_numpy(), coords)
    print(f"[6] Moran's I (log CPM): I={res_log['I']:.4f}  E[I]={res_log['EI']:.4f}  "
          f"z={res_log['z']:.3f}  p_norm={res_log['p_norm']:.3e}  p_perm={res_log['p_perm']:.4f}")
    print(f"    Moran's I (raw)    : I={res_raw['I']:.4f}  z={res_raw['z']:.3f}  "
          f"p_perm={res_raw['p_perm']:.4f}")

    # ---------- 7. 输出 ----------
    tissue.to_csv(os.path.join(OUT_DIR, "TD1_SLPI_spots.csv"), index=False, float_format="%.4f")

    txt = []
    txt.append("SLPI 空间分布独立验证 — GEO GSE189487 / GSM5702473 (TD1, IAC, LUAD Visium)")
    txt.append("=" * 78)
    txt.append(f"基因数: {len(genes)}    barcodes: {len(barcodes)}    组织内 spots: {len(tissue)}")
    txt.append(f"SLPI raw counts: mean={tissue.SLPI_raw.mean():.2f}  median={tissue.SLPI_raw.median():.1f}  "
               f"max={tissue.SLPI_raw.max():.0f}  检出率={ (tissue.SLPI_raw>0).mean()*100:.1f}%")
    txt.append("")
    txt.append(f"Moran's I (KNN k={K_NN}, 行标准化, log1p CPM):")
    txt.append(f"  I = {res_log['I']:.4f}   E[I] = {res_log['EI']:.4f}")
    txt.append(f"  解析 z = {res_log['z']:.3f}, p(单侧正态) = {res_log['p_norm']:.3e}")
    txt.append(f"  置换检验 (n={N_PERM}): p = {res_log['p_perm']:.4f}  (零分布 mean={res_log['perm_mean']:.4f}, sd={res_log['perm_sd']:.4f})")
    txt.append("")
    txt.append(f"Moran's I (raw counts): I = {res_raw['I']:.4f}, z = {res_raw['z']:.3f}, p_perm = {res_raw['p_perm']:.4f}")
    txt.append("")
    verdict = ("SLPI 表达呈显著空间正自相关 (I>0, p<0.05)"
               if res_log['I'] > 0 and res_log['p_perm'] < 0.05
               else "SLPI 表达空间自相关不显著 (p>=0.05)")
    txt.append(f"结论: {verdict}")
    text = "\n".join(txt)
    with open(os.path.join(OUT_DIR, "TD1_SLPI_result.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)

    # ---------- 8. 绘图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    sc = axes[0].scatter(tissue.pxl_col, tissue.pxl_row, c=tissue.SLPI_log_cpm,
                         cmap="YlOrRd", s=14, edgecolors="none")
    axes[0].invert_yaxis(); axes[0].set_aspect("equal")
    axes[0].set_title(f"SLPI log1p(CPM), TD1 (IAC)\nMoran's I = {res_log['I']:.3f}, p_perm = {res_log['p_perm']:.3f}")
    axes[0].set_xlabel("pxl_col"); axes[0].set_ylabel("pxl_row")
    plt.colorbar(sc, ax=axes[0], fraction=0.046)

    axes[1].scatter(tissue.pxl_col, tissue.pxl_row, c=tissue.SLPI_raw,
                    cmap="viridis", s=14, edgecolors="none")
    axes[1].invert_yaxis(); axes[1].set_aspect("equal")
    axes[1].set_title("SLPI raw counts")
    axes[1].set_xlabel("pxl_col"); axes[1].set_ylabel("pxl_row")
    plt.colorbar(axes[1].collections[0], ax=axes[1], fraction=0.046)

    axes[2].hist(tissue.SLPI_log_cpm, bins=40, color="steelblue", edgecolor="black", linewidth=0.4)
    axes[2].set_title(f"SLPI log1p(CPM) distribution\nspots = {len(tissue)}")
    axes[2].set_xlabel("log1p(CPM)"); axes[2].set_ylabel("# spots")

    plt.tight_layout()
    png = os.path.join(OUT_DIR, "TD1_SLPI_spatial_plot.png")
    plt.savefig(png, dpi=170, bbox_inches="tight")
    print(f"[8] 图已保存: {png}")


if __name__ == "__main__":
    main()
