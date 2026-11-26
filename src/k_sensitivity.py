"""P2a: K 敏感性分析 —— 不同 niche 数 K 下重建 CPTAC UCEC 特征并跑蛋白组下游。

目的: 验证「拓扑转移特征主导」结论对 niche 数 K=10 的选择不敏感。
方法: 对 K∈{5,8,15,20} 重新全局 KMeans → 重建参数化特征 → full / only-trans 两配置
      → CPTAC UCEC 蛋白组预测 (top2000, 5-fold×3seeds, scipy p + BH-FDR)。

注: interface/caf 特征依赖 K=10 的语义索引 (niche 1/4/5/6=免疫等), K 变化后簇身份
    重排, 语义特征无定义, 故本分析使用结构一致的特征集:
      abund(K) + ripley(4K) + div(4+K) + trans(K²) + corr(K(K-1)/2)

用法 (每 K 一个进程, 可并行):
  python src/k_sensitivity.py --k 5  --out-root results_vitl14_jbi_pancancer/p2_ksens
  python src/k_sensitivity.py --k 20 --out-root results_vitl14_jbi_pancancer/p2_ksens
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from sklearn.cluster import KMeans

from downstream_omics import (SEED, aggregate_feature_importance,
                              build_morphology_matrix, load_cptac_omics,
                              run_modality, unify_patient_ids)

EMBED_DIR = config.DATA_PROC / "dinov2_vitl14_embeddings_ucec"
OMICS_COHORT = "UCEC"
KS = [5, 8, 10, 15, 20]


# ==============================================================
# 参数化 K 的特征函数 (逻辑照抄 features.py, N_NICHES → K)
# ==============================================================
def ripley_k(probs: np.ndarray, K: int, radii=(1, 2, 3, 4), grid=20) -> np.ndarray:
    n_patches = len(probs)
    if n_patches < grid * grid:
        return np.zeros(K * len(radii))
    k_per_niche_radius = []
    for k in range(K):
        binary = (probs[:, k] > probs[:, k].mean()).astype(int)
        padded = np.zeros(grid * grid, dtype=int)
        padded[:min(n_patches, grid * grid)] = binary[:min(n_patches, grid * grid)]
        grid_2d = padded.reshape(grid, grid)
        coords = np.argwhere(grid_2d == 1)
        n = len(coords)
        if n < 2:
            k_per_niche_radius.extend([0.0] * len(radii))
            continue
        vals = []
        for r in radii:
            count = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        d = np.abs(coords[i] - coords[j]).max()
                        if d <= r:
                            count += 1
            vals.append(count / (n * (n - 1)) if n > 1 else 0.0)
        k_per_niche_radius.extend(vals)
    return np.array(k_per_niche_radius, dtype=np.float32)


def topology_features(argmax: np.ndarray, K: int) -> np.ndarray:
    n = len(argmax)
    if n < 2:
        return np.zeros(K * K, dtype=np.float32)
    t = np.zeros((K, K), dtype=np.float32)
    for i in range(n - 1):
        t[argmax[i], argmax[i + 1]] += 1
    row_sum = t.sum(axis=1, keepdims=True) + 1e-6
    return (t / row_sum).flatten()


def niche_correlations(probs: np.ndarray, K: int) -> np.ndarray:
    feats = []
    for i in range(K):
        for j in range(i + 1, K):
            if probs[:, i].std() > 0 and probs[:, j].std() > 0:
                feats.append(float(np.corrcoef(probs[:, i], probs[:, j])[0, 1]))
            else:
                feats.append(0.0)
    return np.array(feats, dtype=np.float32)


def feature_names_for(K: int) -> list[str]:
    return (
        [f"abund_{k}" for k in range(K)] +
        [f"ripley_{k}_{r}" for k in range(K) for r in range(1, 5)] +
        [f"div_{i}" for i in range(4 + K)] +
        [f"trans_{k}_{l}" for k in range(K) for l in range(K)] +
        [f"corr_{i}_{j}" for i in range(K) for j in range(i + 1, K)]
    )


def extract_one(probs: np.ndarray, argmax: np.ndarray, K: int) -> np.ndarray:
    shannon = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    simpson = 1 - np.sum(probs ** 2, axis=1)
    div = [float(shannon.mean()), float(shannon.std()),
           float(simpson.mean()), float(simpson.std())]
    div += [float(-probs[:, k].mean() * np.log(probs[:, k].mean() + 1e-9)) for k in range(K)]
    parts = [
        probs.mean(axis=0),          # abund: K
        ripley_k(probs, K),          # ripley: 4K
        np.array(div, dtype=np.float32),  # div: 4+K
        topology_features(argmax, K),     # trans: K²
        niche_correlations(probs, K),     # corr: K(K-1)/2
    ]
    return np.concatenate(parts).astype(np.float32)


# ==============================================================
# 主流程
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, choices=KS)
    ap.add_argument("--out-root", default="results_vitl14_jbi_pancancer/p2_ksens")
    ap.add_argument("--targets-top-k", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    K = args.k
    t0 = time.time()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. 载入 UCEC embeddings
    npy_files = sorted(f for f in EMBED_DIR.glob("*.npy") if "_coords" not in f.name)
    assert npy_files, f"No embeddings in {EMBED_DIR}"
    print(f"[load] {len(npy_files)} WSI embeddings from {EMBED_DIR}")

    # 2. 全局 KMeans (与主分析同协议: 5 万点采样, n_init=20, seed=2026)
    rng = np.random.RandomState(SEED)
    sample = []
    per = 50000 // len(npy_files) + 1
    for f in npy_files:
        feats = np.load(f)
        idx = rng.choice(len(feats), size=min(per, len(feats)), replace=False)
        sample.append(feats[idx])
    X_fit = np.concatenate(sample, axis=0)
    print(f"[kmeans] fitting K={K} on {X_fit.shape[0]} patches")
    km = KMeans(n_clusters=K, n_init=config.KMEANS_N_INIT,
                random_state=SEED, verbose=0)
    km.fit(X_fit)
    del X_fit, sample
    print(f"[kmeans] done in {(time.time()-t0)/60:.1f} min, inertia={km.inertia_:.3e}")

    # 3. 逐切片 decompose + 特征
    rows, ids = [], []
    for f in npy_files:
        case_id = f.name.replace(".npy", "")
        embed = np.load(f)
        dists = km.transform(embed)
        inv = 1.0 / (dists + 1e-6)
        probs = (inv / inv.sum(axis=1, keepdims=True)).astype(np.float32)
        argmax = km.predict(embed).astype(np.int16)
        rows.append(extract_one(probs, argmax, K))
        ids.append(case_id)
    feat_df = pd.DataFrame(rows, columns=feature_names_for(K))
    feat_df.insert(0, "case_id", ids)
    feat_df["_cohort"] = OMICS_COHORT
    print(f"[features] {feat_df.shape} in {(time.time()-t0)/60:.1f} min")
    feat_csv = out_root / f"features_k{K}.csv"
    feat_df.to_csv(feat_csv, index=False)

    # 4. 蛋白组 (与主分析同一加载/ID统一路径)
    omics = load_cptac_omics(modalities=["proteomics"], cohort=OMICS_COHORT)
    omics = unify_patient_ids(omics)
    om_df = omics["proteomics"].copy()
    feat_df["case_id"] = feat_df["case_id"].astype(str)
    om_df["case_id"] = om_df["case_id"].astype(str)
    common = set(feat_df["case_id"]) & set(om_df["case_id"])
    feat_df = feat_df[feat_df["case_id"].isin(common)].reset_index(drop=True)
    om_df = om_df[om_df["case_id"].isin(common)].reset_index(drop=True)
    print(f"[align] UCEC 交集 {len(feat_df)} cases")

    # 5. 两配置下游
    idx_of = {c: i for i, c in enumerate(feat_df["case_id"])}
    om_ids = om_df["case_id"].astype(str).tolist()
    for subset in ["full", "only-trans"]:
        X_all, feat_names = build_morphology_matrix(feat_df, feature_subset=subset)
        X_mod = X_all[[idx_of[c] for c in om_ids]]
        out_dir = out_root / f"k{K}_{subset.replace('-', '')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        r_df = run_modality(
            X=X_mod, omics_df=om_df, cov=None, modality_name="proteomics",
            top_k_variance=args.targets_top_k,
            out_csv=str(out_dir / "prot_results.csv"),
            n_seeds=args.seeds, feature_names=feat_names, do_permutation=False)
        if not r_df.empty:
            imp = aggregate_feature_importance(
                X_mod, om_df, target_cols=r_df["target"].tolist(),
                modality_name="proteomics", seed=SEED)
            if not imp.empty:
                imp.to_csv(out_dir / "prot_importance.csv", index=False)
        sig_n = int((r_df["q_value"] < 0.05).sum()) if "q_value" in r_df else 0
        print(f"\n[汇总] tcga_ucec_k{K}_{subset}: 显著 {sig_n}/{len(r_df)} (q<0.05)")

    print(f"\n[DONE] K={K} 总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
