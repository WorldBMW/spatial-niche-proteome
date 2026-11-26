"""
SLPI 空间自相关加固分析:
1. k 敏感性: k in {4,6,8,10,12} 的 Moran's I 与置换 p (log1p CPM + raw)
2. 全转录组基准: 对全部 36,601 个基因计算 Moran's I (k=6, log1p CPM)，
   给出 SLPI 的分位位置与超过 SLPI 的基因数
输出: TD1_SLPI_extended.txt, TD1_genome_MoransI.csv.gz
"""
import gzip
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.spatial import cKDTree

DATA_DIR = r"f:\ai_paper\experiment2_fm_time\data\visium_luad\TD1_IAC"
OUT_DIR = r"f:\ai_paper\experiment2_fm_time\data\visium_luad"
GENE = "SLPI"
RNG_SEED = 42
N_PERM = 1000
KS = [4, 6, 8, 10, 12]


def read_tsv_gz(path, min_cols=1):
    rows = []
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= min_cols:
                rows.append(parts)
    return rows


def build_w(coords, k):
    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=k + 1)
    nbr = idx[:, 1:]
    n = coords.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = nbr.ravel()
    data = np.full(n * k, 1.0 / k)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n)), nbr


def morans_stat(x, W, nbr, n_perm=0, seed=RNG_SEED):
    """x: 1-D 表达值; 返回 I, z, p_norm, p_perm, E[I]"""
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    denom = float((z ** 2).sum())
    if denom <= 0:
        return dict(I=np.nan, z=np.nan, p_norm=np.nan, p_perm=np.nan, EI=-1.0 / (len(x) - 1))
    lag = W.dot(z)
    n = len(x)
    S0 = n  # 行标准化后 S0 = n
    I = float((z * lag).sum() / denom)
    # 权重统计量 S0/S1/S2
    WT = W.T.tocsr()
    B = (W + WT)
    S1 = float(B.multiply(B).sum()) * 0.5
    A2 = np.asarray(W.sum(axis=0)).ravel() + np.asarray(W.sum(axis=1)).ravel()
    S2 = float((A2 ** 2).sum())
    b2 = n * float(((z ** 2) ** 2).sum()) / denom ** 2
    EI = -1.0 / (n - 1)
    vnum = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    vden = (n - 1) * (n - 2) * (n - 3) * S0 ** 2
    var = (vnum - b2 * (n * (n - 1) * S1 - 2 * n * S2 + 6 * S0 ** 2)) / vden
    zscore = (I - EI) / np.sqrt(abs(var)) if var != 0 else np.nan
    from math import erf, sqrt
    p_norm = 2 * (1 - 0.5 * (1 + erf(abs(zscore) / sqrt(2))) if zscore == zscore else np.nan)
    p_perm = np.nan
    if n_perm:
        rng = np.random.default_rng(seed)
        cnt = 0
        for _ in range(n_perm):
            xs = rng.permutation(x)
            zs = xs - xs.mean()
            Is = float((zs * W.dot(zs)).sum() / (zs ** 2).sum())
            if Is >= I:
                cnt += 1
        p_perm = (cnt + 1) / (n_perm + 1)
    return dict(I=I, z=zscore, p_norm=p_norm, p_perm=p_perm, EI=EI)


def main():
    t0 = time.time()
    feats = read_tsv_gz(os.path.join(DATA_DIR, "GSM5702473_TD1_features.tsv.gz"), 3)
    genes = np.array([r[1] for r in feats])
    barcodes = [r[0] for r in read_tsv_gz(os.path.join(DATA_DIR, "GSM5702473_TD1_barcodes.tsv.gz"), 1)]

    pos_rows = []
    with gzip.open(os.path.join(DATA_DIR, "GSM5702473_TD1_tissue_positions_list.csv.gz"), "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= 6:
                pos_rows.append((p[0], int(p[1]), float(p[4]), float(p[5])))
    pos = pd.DataFrame(pos_rows, columns=["barcode", "in_tissue", "pxl_row", "pxl_col"])
    pos = pos[pos.in_tissue == 1].reset_index(drop=True)
    bc_index = {b: i for i, b in enumerate(barcodes)}
    cols = np.array([bc_index[b] for b in pos.barcode])

    print("[load] matrix ...")
    mat = mmread(os.path.join(DATA_DIR, "GSM5702473_TD1_matrix.mtx.gz")).tocsc()
    mat = mat[:, cols]                      # 只保留组织内 spot
    print("[load] matrix shape (genes x tissue spots) =", mat.shape, " nnz =", mat.nnz)
    total = np.asarray(mat.sum(axis=0)).ravel()
    coords = pos[["pxl_col", "pxl_row"]].to_numpy(dtype=float)
    n = coords.shape[0]

    slpi_i = int(np.where(genes == GENE)[0][0])
    raw_slpi = np.asarray(mat[slpi_i, :].todense()).ravel()
    cpm_slpi = np.divide(raw_slpi * 1e4, total, out=np.zeros(n, dtype=float), where=total > 0)
    log_slpi = np.log1p(cpm_slpi)

    lines = []
    lines.append("SLPI 空间自相关加固分析 — GSE189487 / GSM5702473 (TD1, IAC, LUAD Visium)")
    lines.append("=" * 78)
    lines.append("组织内 spot n = %d ; SLPI 检出率 = %.1f%% ; raw mean = %.2f ; max = %.0f"
                 % (n, 100.0 * (raw_slpi > 0).mean(), raw_slpi.mean(), raw_slpi.max()))

    # ---- 1. k 敏感性 ----
    lines.append("")
    lines.append("[1] k 敏感性 (KNN 行标准化, 置换 1000 次)")
    lines.append("  k     I(log1pCPM)  z       p_norm   p_perm   I(raw)   p_perm(raw)")
    kres = {}
    for k in KS:
        W, nbr = build_w(coords, k)
        r_log = morans_stat(log_slpi, W, nbr, n_perm=N_PERM)
        r_raw = morans_stat(raw_slpi, W, nbr, n_perm=N_PERM)
        kres[k] = (r_log, r_raw)
        lines.append("  %-3d   %8.4f  %7.3f  %8.2e  %6.4f  %8.4f  %6.4f"
                     % (k, r_log["I"], r_log["z"], r_log["p_norm"], r_log["p_perm"],
                        r_raw["I"], r_raw["p_perm"]))
        print(lines[-1])

    # ---- 2. 全转录组基准 ----
    W6, nbr6 = build_w(coords, 6)
    inv_total = np.divide(1.0, total, out=np.zeros(n, dtype=float), where=total > 0)
    gene_I = np.full(len(genes), np.nan)
    chunk = 2000
    n_nonzero = 0
    for s in range(0, len(genes), chunk):
        e = min(s + chunk, len(genes))
        X = np.asarray(mat[s:e, :].toarray(), dtype=np.float64)
        cpm = X * 1e4 * inv_total[None, :]
        X = np.log1p(cpm)
        X -= X.mean(axis=1, keepdims=True)
        den = (X ** 2).sum(axis=1)
        lag = (W6.dot(X.T)).T
        num = (X * lag).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            I = np.where(den > 0, num / den, np.nan)
        gene_I[s:e] = I
        n_nonzero += int(np.isfinite(I).sum())
        if (s // chunk) % 5 == 0:
            print("  chunk %d-%d done (%.0f s)" % (s, e, time.time() - t0))
    finite = np.isfinite(gene_I)
    vals = gene_I[finite]
    slpi_val = gene_I[slpi_i]
    pct = 100.0 * (vals < slpi_val).mean()
    n_ge = int((vals >= slpi_val).sum())
    lines.append("")
    lines.append("[2] 全转录组 Moran's I 分布 (k=6, log1p CPM, 可用基因 %d)" % finite.sum())
    lines.append("  中位数 = %.4f   P75 = %.4f   P90 = %.4f   P95 = %.4f   P99 = %.4f   最大 = %.4f"
                 % (np.median(vals), np.percentile(vals, 75), np.percentile(vals, 90),
                    np.percentile(vals, 95), np.percentile(vals, 99), vals.max()))
    lines.append("  负值(I<0) 基因占比 = %.1f%%" % (100.0 * (vals < 0).mean()))
    lines.append("  SLPI: I = %.4f, 分位 = %.1f%%, 超过 SLPI 的基因数 = %d / %d (%.2f%%)"
                 % (slpi_val, pct, n_ge, finite.sum(), 100.0 * n_ge / finite.sum()))
    print(lines[-1])

    df = pd.DataFrame({"gene": genes[finite], "morans_I": gene_I[finite]})
    df.sort_values("morans_I", ascending=False).to_csv(
        os.path.join(OUT_DIR, "TD1_genome_MoransI.csv.gz"), index=False, compression="gzip")

    # 3. 负对照: 随机抽取 50 个基因的 I 与 SLPI 比较
    rng = np.random.default_rng(RNG_SEED)
    pick = rng.choice(np.where(finite)[0], size=min(50, int(finite.sum())), replace=False)
    lines.append("")
    lines.append("[3] 随机 50 个基因的 I: 均值 = %.4f, 中位数 = %.4f, 最大 = %.4f"
                 % (np.nanmean(gene_I[pick]), np.nanmedian(gene_I[pick]), np.nanmax(gene_I[pick])))

    lines.append("")
    lines.append("总耗时 = %.1f s" % (time.time() - t0))
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "TD1_SLPI_extended.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
