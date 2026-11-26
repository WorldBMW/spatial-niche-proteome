# -*- coding: utf-8 -*-
"""
combat_sensitivity.py — ComBat 批次校正敏感性分析

目的: 验证协变量阶梯 (A2) 的结论不依赖是否做 ComBat 批次校正。
做法: 对 CPTAC NSCLC 特征做 ComBat (cohort=LUAD/LUSC 作为 batch) 校正,
      比较校正前后下游蛋白质组命中数变化。

用法:
  python scripts/combat_sensitivity.py
"""
import sys, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
import downstream_omics as do

FEAT_FP = config.ROOT / "results_vitl14_jbi" / "all_features_omics_aligned.csv"
OUT_DIR = config.ROOT / "results_dict_encoder_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(n_proteins=500):
    """加载 CPTAC NSCLC 特征 + 蛋白质组 (内存友好)."""
    feat = pd.read_csv(FEAT_FP, low_memory=False)
    feat["cohort"] = feat["case_id"].str[:3]

    omics_dir = config.DATA_RAW / "CPTAC" / "omics"
    parts = []
    for coh in ["luad", "lusc"]:
        fp = omics_dir / f"{coh}_proteomics.csv"
        if fp.exists():
            with open(fp, encoding="latin-1") as f:
                header = f.readline().strip().split(",")
            usecols = [0] + list(range(1, min(n_proteins + 1, len(header))))
            df = pd.read_csv(fp, encoding="latin-1", usecols=usecols,
                             on_bad_lines="skip", nrows=200, low_memory=False)
            df["_cohort"] = coh.upper()
            parts.append(df)
    prot = pd.concat(parts, ignore_index=True, join="outer") if parts else pd.DataFrame()
    prot["case_id"] = prot["patient_id"].astype(str).str.strip()

    # 对齐
    feat = feat[feat["case_id"].isin(prot["case_id"])]
    prot = prot[prot["case_id"].isin(feat["case_id"])]
    return feat, prot


def run_downstream(feat_df, prot_df, n_splits=5, n_perm=200):
    """跑 PCA 降维 + 单变量相关 + 置换检验, 返回显著命中数."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr

    # 只取数值列
    feat_num = feat_df.select_dtypes(include=[np.number])
    X = feat_num.values.astype(float)
    # 处理 NaN (列均值填充)
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_means, idx[1])
    case_ids = feat_df["case_id"].values
    prot_num = prot_df.set_index("case_id").select_dtypes(include=[np.number])

    # PCA 降维到 10 个主成分
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_comp = min(10, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=2026)
    X_pca = pca.fit_transform(X_scaled)

    targets = []
    rng = np.random.RandomState(42)
    for col in prot_num.columns:
        y = prot_num[col].reindex(case_ids).values.astype(float)
        valid = ~np.isnan(y)
        if valid.sum() < 30:
            continue
        X_v = X_pca[valid]; y_v = y[valid]
        # 用 PC1 与蛋白质的相关系数作为检验统计量
        obs_stat, _ = pearsonr(X_v[:, 0], y_v)
        # 置换
        perm_stats = []
        for _ in range(n_perm):
            yp = rng.permutation(y_v)
            rp, _ = pearsonr(X_v[:, 0], yp)
            perm_stats.append(abs(rp))
        p_perm = np.mean([1 for ps in perm_stats if ps >= abs(obs_stat)])
        targets.append({"protein": col, "r": round(obs_stat, 4),
                        "p_perm": round(p_perm, 4),
                        "significant": p_perm < 0.05})
    return pd.DataFrame(targets)


def main():
    print("=" * 60)
    print("ComBat 批次校正敏感性分析")
    print("=" * 60)

    feat, prot = load_data(n_proteins=500)
    print(f"[data] feat={feat.shape}, prot={prot.shape}")
    print(f"cohorts: {feat['cohort'].value_counts().to_dict()}")

    # --- 无 ComBat (原始) ---
    print("\n--- 无 ComBat ---")
    res_raw = run_downstream(feat, prot)
    n_raw = int(res_raw["significant"].sum())
    print(f"命中数: {n_raw}/{len(res_raw)}")

    # --- ComBat 校正 ---
    print("\n--- 批次校正 (batch=cohort, mean-centering) ---")
    try:
        feat_num = feat.drop(columns=["case_id", "cohort"], errors="ignore")
        feat_num = feat_num.select_dtypes(include=[np.number])
        X = feat_num.values.astype(float)
        batch = feat["cohort"].values

        # 方法1: pycombat ComBat (若失败则降级为 mean-centering)
        X_corrected = None
        try:
            from pycombat import Combat
            combat = Combat()
            X_corrected = combat.fit_transform(X, batch)
            if np.isnan(X_corrected).any():
                raise ValueError("ComBat 产生 NaN")
            method = "ComBat"
        except Exception as e:
            print(f"  pycombat 失败 ({e}), 降级为 mean-centering")
            # 方法2: 批次均值中心化 (每个 batch 内去均值, 再加全局均值)
            X_corrected = X.copy()
            global_mean = X.mean(axis=0)
            for b in np.unique(batch):
                mask = batch == b
                batch_mean = X[mask].mean(axis=0)
                X_corrected[mask] = X[mask] - batch_mean + global_mean
            method = "mean-centering"

        feat_c = feat.copy()
        for i, col in enumerate(feat_num.columns):
            feat_c[col] = X_corrected[:, i]
        res_combat = run_downstream(feat_c, prot)
        n_combat = int(res_combat["significant"].sum())
        print(f"  校正方法: {method}")
        print(f"  命中数: {n_combat}/{len(res_combat)}")

        # 比较
        merged = res_raw.merge(res_combat, on="protein", suffixes=("_raw", "_combat"))
        n_both = int(((merged["significant_raw"]) & (merged["significant_combat"])).sum())
        n_raw_only = int((merged["significant_raw"] & ~merged["significant_combat"]).sum())
        n_combat_only = int((~merged["significant_raw"] & merged["significant_combat"]).sum())
        print(f"\n重叠: {n_both} | 仅原始: {n_raw_only} | 仅ComBat: {n_combat_only}")

        result = {
            "raw_hits": n_raw,
            "combat_hits": n_combat,
            "overlap": n_both,
            "raw_only": n_raw_only,
            "combat_only": n_combat_only,
            "jaccard": n_both / max(n_raw + n_combat - n_both, 1),
        }
    except Exception as e:
        print(f"ComBat 失败: {e}")
        import traceback; traceback.print_exc()
        result = {"error": str(e)}

    print("\n" + "=" * 60)
    print("ComBat 敏感性结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    out = OUT_DIR / "combat_sensitivity.json"
    import json
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果保存到: {out}")


if __name__ == "__main__":
    main()
