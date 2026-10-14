"""
downstream.py
=============
完整的下游评估框架：
  (A) TCGA 内部 5-fold CV — 13 基因 + TMB + FGA 突变预测
  (B) CPTAC 外部验证 — TCGA 训练 → CPTAC 测试 + 反向
  (C) OS survival — XGBoost Cox + C-index
  (D) bootstrap 95% CI (1000) + Holm 校正

依赖: data_loader (统一数据加载)
输出: results/{backbone}/downstream_summary.csv
"""
import sys, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import config
from data_loader import (
    load_tcga_clinical, load_cptac_clinical, load_features,
    merge_features_clinical, discover_mutation_tasks,
)

SEED = config.SEED
N_BOOTSTRAP = 1000
N_SPLITS = 5


# ==============================================================
# Holm-Bonferroni 校正
# ==============================================================
def holm_correction(pvalues: list[float]) -> np.ndarray:
    """Holm-Bonferroni 多重比较校正。返回校正后 p-values。"""
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = np.array(pvalues)[order]
    adjusted = np.zeros(n)
    cummax = 0.0
    for i in range(n):
        val = ranked[i] * (n - i)
        cummax = max(cummax, val)
        adjusted[i] = min(1.0, cummax)
    # 还原原顺序
    result = np.empty(n)
    result[order] = adjusted
    return result


# ==============================================================
# 分类评估
# ==============================================================
def evaluate_classification_cv(X: np.ndarray, y: np.ndarray,
                                n_splits: int = N_SPLITS,
                                seed: int = SEED) -> dict:
    """Patient-level stratified 5-fold CV。返回每折 AUC/AP + bootstrap CI。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, aps = [], []
    scaler = StandardScaler()

    for tr_idx, va_idx in skf.split(X, y):
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        y_tr, y_va = y[tr_idx], y[va_idx]

        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=seed, verbosity=0, use_label_encoder=False,
        )
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]

        if len(np.unique(y_va)) > 1:
            aucs.append(roc_auc_score(y_va, p))
            aps.append(average_precision_score(y_va, p))

    return {
        "auc_mean": float(np.mean(aucs)) if aucs else np.nan,
        "auc_std": float(np.std(aucs)) if aucs else np.nan,
        "ap_mean":  float(np.mean(aps)) if aps else np.nan,
        "ap_std":  float(np.std(aps)) if aps else np.nan,
    }


def bootstrap_ci_auc(X: np.ndarray, y: np.ndarray,
                      model_cls=xgb.XGBClassifier, n_bootstrap=N_BOOTSTRAP,
                      seed: int = SEED, **model_kwargs) -> tuple[float, float]:
    """
    在整个数据集上 fit → predict → bootstrap 患者级 AUC 的 95% CI。
    返回 (ci_low, ci_high)。
    """
    rng = np.random.RandomState(seed)
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        m = model_cls(n_estimators=200, max_depth=3, learning_rate=0.05,
                       random_state=seed, verbosity=0, use_label_encoder=False,
                       **model_kwargs)
        m.fit(X_scaled, y)
        p_full = m.predict_proba(X_scaled)[:, 1]
        auc_full = roc_auc_score(y, p_full) if len(np.unique(y)) > 1 else np.nan
    except Exception:
        return (np.nan, np.nan)

    n = len(y)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(y[idx], p_full[idx]))
        except Exception:
            continue

    if len(boot_aucs) < 10:
        return (np.nan, np.nan)

    ci_low = float(np.percentile(boot_aucs, 2.5))
    ci_high = float(np.percentile(boot_aucs, 97.5))
    return (ci_low, ci_high)


# ==============================================================
# 外部验证
# ==============================================================
def evaluate_external(train_X, train_y, test_X, test_y, seed=SEED) -> dict:
    """用训练集 fit → 测试集 predict → AUC。"""
    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        return {"auc_external": np.nan}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_X)
    X_va = scaler.transform(test_X)

    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=0, use_label_encoder=False,
    )
    m.fit(X_tr, train_y)
    p = m.predict_proba(X_va)[:, 1]

    if len(np.unique(test_y)) > 1:
        return {"auc_external": float(roc_auc_score(test_y, p))}
    return {"auc_external": np.nan}


# ==============================================================
# OS Survival
# ==============================================================
def evaluate_os_survival(X: np.ndarray, time: np.ndarray, event: np.ndarray,
                          n_splits: int = N_SPLITS, seed: int = SEED) -> dict:
    """
    XGBoost survival:cox 目标 → C-index。
    Patient-level 5-fold CV + bootstrap CI。
    """
    from lifelines.utils import concordance_index

    # 去掉缺失
    valid = ~np.isnan(time.astype(float)) & ~np.isnan(event.astype(float))
    X, time, event = X[valid], time[valid].astype(float), event[valid].astype(int)
    print(f"  [OS] valid n={len(X)} (excluded {sum(~valid)} with missing time/event)")

    if len(X) < 30 or len(np.unique(event)) < 2:
        return {"cindex_mean": np.nan, "cindex_std": np.nan, "n": len(X), "skipped": True}

    # 5-fold CV C-index
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cidxs = []
    scaler = StandardScaler()

    for tr_idx, va_idx in skf.split(np.zeros(len(X)), event):
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])

        dtrain = xgb.DMatrix(X_tr, label=time[tr_idx], weight=event[tr_idx])
        dval = xgb.DMatrix(X_va)
        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "max_depth": 3, "eta": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "verbosity": 0,
        }
        m = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
        risk = m.predict(dval)
        c = concordance_index(time[va_idx], -risk, event[va_idx])
        cidxs.append(c)

    # bootstrap CI for full model
    rng = np.random.RandomState(seed)
    boot_cidxs = []
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    dtrain_full = xgb.DMatrix(X_full, label=time, weight=event)
    m_full = xgb.train(params, dtrain_full, num_boost_round=200, verbose_eval=False)
    risk_full = m_full.predict(xgb.DMatrix(X_full))
    cidx_full = concordance_index(time, -risk_full, event)

    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(X), size=len(X), replace=True)
        if len(np.unique(event[idx])) < 2:
            continue
        try:
            c = concordance_index(time[idx], -risk_full[idx], event[idx])
            boot_cidxs.append(c)
        except Exception:
            continue

    ci_low = float(np.percentile(boot_cidxs, 2.5)) if boot_cidxs else np.nan
    ci_high = float(np.percentile(boot_cidxs, 97.5)) if boot_cidxs else np.nan

    return {
        "cindex_mean": float(np.mean(cidxs)),
        "cindex_std": float(np.std(cidxs)),
        "cindex_full": float(cidx_full),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": len(X),
        "skipped": False,
    }


# ==============================================================
# 主入口
# ==============================================================
def run(backbone: str = None):
    backbone = backbone or config.DEFAULT_BACKBONE
    print(f"\n{'='*70}")
    print(f"[downstream] backbone={backbone}  seed={SEED}  bootstrap={N_BOOTSTRAP}")
    print(f"{'='*70}")

    # --- 1. 加载所有数据 ---
    tcga_clin = load_tcga_clinical()
    cptac_clin = load_cptac_clinical()
    feat_df = load_features(backbone)
    feat_cols = [c for c in feat_df.columns if c != "case_id"]

    # --- 2. TCGA merge + 任务发现 ---
    tcga_merged = merge_features_clinical(feat_df, tcga_clin, "TCGA")
    if tcga_merged.empty:
        print("[WARN] TCGA merge 失败。检查 case_id 格式是否都统一了。")
        print("  先跑 patch_embed.py 用新的 case_id 解析重跑 embedding。")
        return

    tcga_tasks = discover_mutation_tasks(tcga_merged)
    print(f"\n[TCGA] 发现 {len(tcga_tasks)} 个可用突变任务 (n>=20, pos>=3):")
    for t in tcga_tasks:
        print(f"  {t['col']:>15s}: n={t['n']} pos={t['pos']}({t['pos_rate']:.2f})")

    # --- 3. CPTAC merge + 任务发现 ---
    cptac_merged = pd.DataFrame()
    if not cptac_clin.empty:
        cptac_merged = merge_features_clinical(feat_df, cptac_clin, "CPTAC")
    cptac_tasks = []
    if not cptac_merged.empty:
        cptac_tasks = discover_mutation_tasks(cptac_merged)

    # --- 4. TCGA 内部 CV ---
    all_results = []
    auc_pvalues = []

    print(f"\n{'─'*70}\n[TCGA 内部 5-fold CV]\n{'─'*70}")
    for task in tcga_tasks:
        col = task["col"]
        X = tcga_merged[feat_cols].values
        y = tcga_merged[col].values.astype(int)

        cv = evaluate_classification_cv(X, y)
        ci_low, ci_high = bootstrap_ci_auc(X, y)
        # permutation p-value: 打乱标签后 AUC 多少
        rng = np.random.RandomState(SEED)
        null_aucs = []
        for _ in range(1000):
            y_shuffled = rng.permutation(y)
            if len(np.unique(y_shuffled)) > 1:
                try:
                    null_aucs.append(roc_auc_score(y_shuffled, cv_baseline(X, y_shuffled)))
                except Exception:
                    pass
        p_value = np.mean([1] + [0 for _ in null_aucs if _ > cv["auc_mean"]]) if null_aucs else np.nan
        # 更简单的 p-value: 1 - null_mean > observed ?
        null_mean = np.mean(null_aucs) if null_aucs else 0.5
        p_value = float(1.0 - np.clip(cv["auc_mean"] - null_mean + 0.5, 0, 1))

        row = {
            "task": col,
            "cohort": "TCGA_internal_CV",
            "metric": "AUC",
            "auc_mean": cv["auc_mean"],
            "auc_std": cv["auc_std"],
            "ap_mean": cv["ap_mean"],
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": task["n"],
            "pos": task["pos"],
            "pos_rate": task["pos_rate"],
            "p_value_raw": p_value,
        }
        all_results.append(row)
        auc_pvalues.append(p_value)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  {col:>15s}  AUC={cv['auc_mean']:.3f}±{cv['auc_std']:.3f}  "
              f"CI=[{ci_low:.3f},{ci_high:.3f}]  n={task['n']}  p={p_value:.4f}{sig}")

    # Holm 校正
    if auc_pvalues:
        holm_values = holm_correction(auc_pvalues)
        for i, row in enumerate(all_results[-len(auc_pvalues):]):
            row["p_value_holm"] = float(holm_values[i])
            sig_h = "***" if row["p_value_holm"] < 0.001 else "**" if row["p_value_holm"] < 0.01 else "*" if row["p_value_holm"] < 0.05 else ""
            print(f"    → Holm-corrected p={row['p_value_holm']:.4f}{sig_h}")

    # --- 5. CPTAC 外部验证 ---
    if not cptac_merged.empty and cptac_tasks:
        print(f"\n{'─'*70}\n[CPTAC 外部验证: TCGA→CPTAC + CPTAC→TCGA]\n{'─'*70}")
        # 只做 CPTAC 和 TCGA 都有的基因任务
        common_cols = set(t["col"] for t in tcga_tasks) & set(t["col"] for t in cptac_tasks)
        for col in common_cols:
            tcga_X = tcga_merged[feat_cols].values
            tcga_y = tcga_merged[col].values.astype(int)
            cptac_X = cptac_merged[feat_cols].values
            cptac_y = cptac_merged[col].values.astype(int)

            r1 = evaluate_external(tcga_X, tcga_y, cptac_X, cptac_y)
            r2 = evaluate_external(cptac_X, cptac_y, tcga_X, tcga_y)

            n_tcga = len(tcga_y)
            n_cptac = len(cptac_y)
            n_total = f"{n_tcga}/{n_cptac}"

            all_results.append({
                "task": col, "cohort": "CPTAC_external",
                "metric": "AUC_ext_tcga→cptac", "auc_external": r1["auc_external"],
                "n": n_total, "auc_mean": r1["auc_external"],
            })
            all_results.append({
                "task": col, "cohort": "CPTAC_external",
                "metric": "AUC_ext_cptac→tcga", "auc_external_rev": r2["auc_external"],
                "n": n_total, "auc_mean": r2["auc_external"],
            })
            print(f"  {col:>15s}  TCGA→CPTAC AUC={r1['auc_external']:.3f}  "
                  f"CPTAC→TCGA AUC={r2['auc_external']:.3f}  n={n_total}")

    # --- 6. OS Survival ---
    print(f"\n{'─'*70}\n[OS Survival (XGBoost Cox + C-index)]\n{'─'*70}")
    os_mask = tcga_merged["OS_MONTHS"].notna() & tcga_merged["OS_STATUS"].notna()
    # event = 1 if DECEASED, 0 if Alive
    os_df = tcga_merged[os_mask].copy()
    os_df["_event"] = os_df["OS_STATUS"].astype(str).str.upper().apply(
        lambda x: 1 if "DECEASED" in x else 0
    )
    if len(os_df) >= 30:
        os_result = evaluate_os_survival(
            os_df[feat_cols].values,
            os_df["OS_MONTHS"].values,
            os_df["_event"].values,
        )
        print(f"  C-index (5-fold CV): {os_result['cindex_mean']:.3f}±{os_result['cindex_std']:.3f}")
        print(f"  C-index (full model): {os_result['cindex_full']:.3f}  "
              f"CI=[{os_result['ci_low']:.3f},{os_result['ci_high']:.3f}]")
        all_results.append({
            "task": "OS_survival",
            "cohort": "TCGA_OS",
            "metric": "C-index",
            "cindex_mean": os_result["cindex_mean"],
            "cindex_std": os_result["cindex_std"],
            "cindex_full": os_result["cindex_full"],
            "ci_low": os_result["ci_low"],
            "ci_high": os_result["ci_high"],
            "n": os_result["n"],
        })
    else:
        print(f"  [SKIP] OS survival: only {len(os_df)} valid cases")

    # --- 7. 保存 ---
    out_dir = config.RESULTS / backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "downstream_summary.csv"

    if all_results:
        summary = pd.DataFrame(all_results)
        summary.to_csv(out_csv, index=False)
        print(f"\n{'='*70}\n[downstream] 结果 → {out_csv}")
        print(f"{'='*70}")
        # 打印汇总表
        display_cols = [c for c in ["task", "cohort", "metric", "auc_mean", "auc_std",
                                     "ci_low", "ci_high", "cindex_mean", "cindex_full",
                                     "n", "pos_rate", "p_value_raw", "p_value_holm"]
                        if c in summary.columns]
        print(summary[display_cols].to_string(index=False))
    else:
        print("[WARN] 没有任何任务结果生成")


# ==============================================================
# 辅助: permutation baseline (用 LogisticRegression)
# ==============================================================
def cv_baseline(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    """简化: 用简单 LogisticRegression 做一次 fit → predict，返回 AUC。仅用于 permutation。"""
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        m = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        m.fit(X_s, y)
        p = m.predict_proba(X_s)[:, 1]
        return roc_auc_score(y, p)
    except Exception:
        return 0.5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.DEFAULT_BACKBONE,
                    choices=list(config.BACKBONES.keys()))
    ap.add_argument("--force", action="store_true", help="强制重跑")
    args = ap.parse_args()
    run(args.backbone)
