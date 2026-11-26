# -*- coding: utf-8 -*-
"""
permutation_resolution.py — 置换次数敏感性分析 (ElasticNet 折外预测统计量)

目的: 用与主分析一致的统计量 (ElasticNet 5-fold OOF Pearson r) 做置换检验,
      核对经验 p 值与解析 p 值的一致性。
做法: 对 frozen_cov 结果中 top N 显著蛋白, 用 ElasticNet OOF r 做 n_perm 次置换
      (每次置换重拟合模型), 计算经验 p。
"""
import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from downstream_omics import load_morphology_features, load_cptac_omics, unify_patient_ids, build_morphology_matrix
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

FEAT_FP = config.ROOT / "results_vitl14_jbi" / "all_features_omics_aligned.csv"
RESULTS_FP = config.ROOT / "results_vitl14_jbi_r4" / "dn_r4_frozen_cov" / "ALL_proteomics_results.csv"
OUT_DIR = config.ROOT / "results_dict_encoder_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def permutation_test_enet(X, y, seed=2026, n_perm=2000):
    """ElasticNet 折外预测置换检验: 统计量 = pooled OOF Pearson r, 每次置换重拟合."""
    rng = np.random.RandomState(seed)
    outer = KFold(n_splits=5, shuffle=True, random_state=seed)

    def model_fn(X_tr, y_tr, X_va):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(np.nan_to_num(X_tr, nan=0))
        X_va_s = sc.transform(np.nan_to_num(X_va, nan=0))
        en = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=seed)
        en.fit(X_tr_s, y_tr)
        return en.predict(X_va_s)

    # 观测统计量: pooled OOF Pearson r
    preds, ytrue = [], []
    for tr_idx, va_idx in outer.split(X):
        y_pred = model_fn(X[tr_idx], y[tr_idx], X[va_idx])
        preds.extend(y_pred)
        ytrue.extend(y[va_idx])
    obs_r, _ = pearsonr(ytrue, preds)

    # 置换: 每次重拟合
    null_rs = np.empty(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        p_perm, yt_perm = [], []
        for tr_idx, va_idx in outer.split(X):
            y_pred = model_fn(X[tr_idx], y_shuf[tr_idx], X[va_idx])
            p_perm.extend(y_pred)
            yt_perm.extend(y_shuf[va_idx])
        null_rs[i], _ = pearsonr(yt_perm, p_perm)

    p = (np.sum(np.abs(null_rs) >= abs(obs_r)) + 1) / (n_perm + 1)
    return float(p), float(obs_r), null_rs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-top", type=int, default=20, help="top N 显著目标")
    ap.add_argument("--n-perm", type=int, default=2000, help="置换次数")
    args = ap.parse_args()

    print("=" * 60)
    print(f"置换检验 (ElasticNet OOF r, n_perm={args.n_perm})")
    print("=" * 60)

    # 加载已有结果, 取 top N 显著目标
    res = pd.read_csv(RESULTS_FP)
    top = res.nsmallest(args.n_top, "permutation_p")[["target", "pearson", "permutation_p", "q_value"]]
    print(f"\nTop {args.n_top} targets (by analytic p-value):")
    print(top.to_string())

    # 加载数据
    omics = load_cptac_omics(config.DATA_RAW / "CPTAC" / "omics",
                             modalities=["proteomics"], cohort="ALL")
    omics = unify_patient_ids(omics)
    prot = omics["proteomics"].set_index("patient_id")
    feat_df = load_morphology_features(str(FEAT_FP), mock=False,
                                        real_roster=omics.get("roster"))
    case_ids = feat_df["case_id"].astype(str).tolist()
    X_all, feat_names = build_morphology_matrix(feat_df, feature_subset="full")

    # 对每个 top target 跑置换
    results = []
    for _, row in top.iterrows():
        target = row["target"]
        if target not in prot.columns:
            print(f"  {target}: not found in proteomics, skipping")
            continue
        y = prot[target].reindex(case_ids).values.astype(float)
        mask = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y)
        if mask.sum() < 30:
            print(f"  {target}: only {mask.sum()} valid, skipping")
            continue
        X_ok = X_all[mask]; y_ok = y[mask]
        t0 = time.time()
        p_perm, obs_r, null_rs = permutation_test_enet(X_ok, y_ok, n_perm=args.n_perm)
        elapsed = time.time() - t0
        results.append({
            "target": target,
            "analytic_p": row["permutation_p"],
            "perm_p": p_perm,
            "obs_r": obs_r,
            "null_r_mean": float(np.mean(null_rs)),
            "null_r_std": float(np.std(null_rs)),
            "n_valid": int(mask.sum()),
            "elapsed_s": round(elapsed, 1),
        })
        print(f"  {target}: analytic_p={row['permutation_p']:.2e}, "
              f"perm_p={p_perm:.2e}, obs_r={obs_r:.4f}, "
              f"null_r=mean±std={np.mean(null_rs):.4f}±{np.std(null_rs):.4f} "
              f"({elapsed:.0f}s)")

    df = pd.DataFrame(results)
    out = OUT_DIR / f"permutation_resolution_enet_{args.n_perm}.csv"
    df.to_csv(out, index=False)
    print(f"\n结果保存到: {out}")

    # 汇总
    if len(df) > 0:
        n_sig_analytic = (df["analytic_p"] < 0.05).sum()
        n_sig_perm = (df["perm_p"] < 0.05).sum()
        print(f"\n汇总: {n_sig_analytic}/{len(df)} analytic p<0.05, "
              f"{n_sig_perm}/{len(df)} perm p<0.05")
        if n_sig_perm == len(df):
            print("结论: 所有 top 目标在置换检验下仍显著, 经验 p 与解析 p 一致")
        elif n_sig_perm == 0:
            print("结论: 所有 top 目标在置换检验下均不显著, 解析 p 严重偏乐观")
        else:
            print(f"结论: {n_sig_perm}/{len(df)} 目标在置换检验下仍显著")


if __name__ == "__main__":
    main()
