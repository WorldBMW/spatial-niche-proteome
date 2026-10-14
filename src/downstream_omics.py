"""
downstream_omics.py —— 形态学特征 → CPTAC 多组学连续标签回归

核心目标: 量化 262 维空间生态特征 (niche abundance + Ripley K + 界面 + 纤维化 + 多样性 + 拓扑)
         对蛋白质组/磷酸化蛋白质组/转录组的预测力, 并识别关键生态特征。

方法学 (解决旧版 downstream.py 的 8 项硬伤):
  1. 嵌套 5-fold × 3 seed, 避免 in-sample bootstrap CI=1.0 假阳性
  2. 回归而非分类 (Pearson r / R² / NRMSE), 连续标签合理评估
  3. BH-FDR 多重比较校正 (q<0.05 阈值)
  4. 协变量残差化 (形态特征 → age/sex/stage/smoking 取残差 → 纯净形态信号)
  5. LUAD/LUSC 分层分析 (各癌种独立 BH-FDR)
  6. 偏相关 (形态 × target 控制协变量)
  7. 跨 target 特征重要性聚合 (识别关键生态特征)
  8. survival 附加模块 (形态特征 + OS Cox, 多因素校正)

输入:
  - features_omics_aligned.csv (run_cptac.py --step align 输出)
    列: case_id, cohort(C3L-/C3N-), 262 维形态特征
  - CPTAC omics 11 csv (data/raw/CPTAC/omics/)

用法:
  python src/downstream_omics.py --features data/features/dinov2_vitl14_features_cptac/all_features_omics_aligned.csv
  python src/downstream_omics.py --features <feat_csv> --cohort LUAD   # 分层
  python src/downstream_omics.py --features <feat_csv> --targets top500  # 只跑 top500 高方差 target (CPU 友好)
  python src/downstream_omics.py --features <feat_csv> --mock           # 合成特征 sanity check
"""
import sys, os, argparse, warnings, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, KFold, cross_val_predict,
                                    ParameterGrid)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import config

SEED = config.SEED if hasattr(config, "SEED") else 2026
N_SEEDS = 3
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 3  # 内层 fold 从 5 降到 3, 提速 40%
BOOTSTRAP_N = 2000

# ==============================================================
# 1. 数据加载层
# ==============================================================

def robust_read_csv(fp: str | Path) -> pd.DataFrame:
    """
    读 CPTAC CSV, 容错行内字段数不一致 + 重复列名 + 空值.
    快速方案: csv 数 header → pandas header=0 + skip + usecols → 去重 + 数值转换.
    """
    fp = str(fp)
    import csv as _csv
    _csv.field_size_limit(10 * 1024 * 1024)
    # 1. csv 数 header 字段数 + 检查重复列名
    with open(fp, encoding="latin-1") as f:
        reader = _csv.reader(f)
        header = next(reader)
    ncols = len(header)
    # 去重列名 (保留首次出现)
    seen = set()
    keep_mask = []
    dup_count = 0
    for h in header:
        if h not in seen:
            seen.add(h)
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            dup_count += 1
    keep_indices = [i for i, m in enumerate(keep_mask) if m]
    # 2. pandas 快速读: header=0, on_bad_lines='skip', usecols=前 ncols
    df = pd.read_csv(fp, encoding="latin-1", header=0,
                     on_bad_lines="skip", usecols=range(ncols))
    # 3. 切到不重复的列
    if dup_count > 0:
        df = df.iloc[:, keep_indices]
        df.columns = [header[i] for i in keep_indices]
    # 4. 空字符串 → NaN (pandas header=0 已自动推断 dtypes, 不需再 to_numeric)
    df = df.replace("", np.nan)
    # 5. patient_id 保持 str
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        df.loc[df["patient_id"].isin(["nan", "None", ""]), "patient_id"] = np.nan
    print(f"[robust_read] {fp.split(chr(92))[-1]}: {df.shape}, dup={dup_count}")
    return df


def load_cptac_omics(omics_dir: str | Path = None,
                     modalities: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """
    加载 CPTAC-LUAD/LUSC 多组学数据, 返回 dict:
      {
        "proteomics":        df (patient_id × 基因),
        "phosphoproteomics": df (patient_id × 基因|位点|肽|ensembl),
        "transcriptomics":   df (patient_id × 基因),
        "clinical_luad":     df,
        "clinical_lusc":     df,
        "somatic_mutation":  df,
        "roster":            df,
      }
    """
    d = Path(omics_dir) if omics_dir else config.DATA_RAW / "CPTAC" / "omics"
    assert d.exists(), f"omics 目录不存在: {d}"

    omics = {}
    # 只加载指定模态 (避免一次性加载全部 180k+ 列挤爆内存)
    omics_names = modalities if modalities else ["proteomics", "phosphoproteomics", "transcriptomics"]
    for name in omics_names:
        parts = []
        for cohort in ["luad", "lusc"]:
            fp = d / f"{cohort}_{name}.csv"
            if fp.exists():
                df = robust_read_csv(fp)
                df["_cohort"] = cohort.upper()
                parts.append(df)
        if parts:
            # join='outer' 取列名并集, 因为 LUAD/LUSC 蛋白名有重叠
            omics[name] = pd.concat(parts, ignore_index=True, join="outer")

    for name in ["clinical", "somatic_mutation"]:
        parts = []
        for cohort in ["luad", "lusc"]:
            fp = d / f"{cohort}_{name}.csv"
            if fp.exists():
                df = robust_read_csv(fp)
                df["_cohort"] = cohort.upper()
                parts.append(df)
        if parts:
            omics[name] = pd.concat(parts, ignore_index=True, join="outer")

    roster_fp = d / "patient_roster.csv"
    omics["roster"] = robust_read_csv(roster_fp) if roster_fp.exists() else pd.DataFrame()

    print(f"[omics] 加载完成: {list(omics.keys())}")
    for k, v in omics.items():
        if isinstance(v, pd.DataFrame):
            # 显示 patient_id 列前 3 值验证
            if "patient_id" in v.columns:
                print(f"  {k}: {v.shape}, patient_id head: {v['patient_id'].tolist()[:3]}")
            else:
                print(f"  {k}: {v.shape}")
    return omics


def unify_patient_ids(omics: dict[str, pd.DataFrame],
                      ws_ids: list[str] | None = None) -> dict:
    """
    把 legacy numeric patient_id (11LU013/11LU016...) 映射到 C3L-/C3N- 格式。
    CPTAC 2020 Cell 原始发现队列前 4 例用了内部编号, 后续全部用 C3L-/C3N-。
    从 roster 里找 numeric→C3L- 的映射 (如果 roster 有 numeric 行且能 match)。
    如果 roster 里找不到 → 保留原 ID, 下游 merge 时自动 inner join 排除。
    """
    mapping = {}  # numeric → C3L-/C3N-
    roster = omics.get("roster", pd.DataFrame())
    if not roster.empty:
        # 看 roster 里 numeric ID 能不能被 omics 表里的 C3L- 行对上
        numeric_ids = roster[roster["patient_id"].str.match(r"^\d{2}LU\d{3}$")]["patient_id"].tolist()
        c3_ids = roster[roster["patient_id"].str.match(r"^C3[LN]-")]["patient_id"].tolist()
        if numeric_ids:
            print(f"[ID unify] numeric IDs: {numeric_ids}")
        if c3_ids:
            print(f"[ID unify] C3L-/C3N- IDs: {len(c3_ids)}")

    for k in ["proteomics", "phosphoproteomics", "transcriptomics", "clinical", "somatic_mutation"]:
        if k in omics and "patient_id" in omics[k].columns:
            df = omics[k].copy()
            df["case_id"] = df["patient_id"].astype(str)
            df.loc[~df["case_id"].str.match(r"^C3[LN]-"), "case_id"] = df.loc[
                ~df["case_id"].str.match(r"^C3[LN]-"), "patient_id"
            ].astype(str)
            omics[k] = df

    return omics


def load_morphology_features(features_csv: str | Path,
                             mock: bool = False,
                             n_samples: int = 200,
                             n_features: int = 262,
                             real_roster: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    加载形态特征 CSV。
    mock=True 时生成合成数据 (CPU 上跑 sanity check 用)。
    若传入 real_roster, 从中取真实 C3L-/C3N- ID 作为 case_id。
    """
    if mock:
        np.random.seed(SEED)
        if real_roster is not None and not real_roster.empty:
            real_ids = [cid for cid in real_roster["patient_id"].astype(str)
                        if cid.startswith("C3L-") or cid.startswith("C3N-")]
            real_ids = sorted(real_ids)
            # 等分到 LUAD / LUSC
            luad_ids = [cid for cid in real_ids if cid.startswith("C3L-")][:n_samples//2]
            lusc_ids = [cid for cid in real_ids if cid.startswith("C3N-")][:n_samples//2]
            ids = (luad_ids + lusc_ids)
            n_samples = len(ids)
        else:
            ids = [f"C3L-{i:05d}" for i in range(1, n_samples//2 + 1)] + \
                  [f"C3N-{i:05d}" for i in range(1000, 1000 + n_samples//2)]

        feat = np.random.randn(n_samples, n_features)
        df = pd.DataFrame(feat, columns=[f"f{i}" for i in range(n_features)])
        cohorts = ["LUAD"]*sum(1 for i in ids if i.startswith("C3L-")) + \
                  ["LUSC"]*sum(1 for i in ids if i.startswith("C3N-"))
        df.insert(0, "case_id", ids)
        df.insert(1, "cohort", cohorts)
        print(f"[mock features] {df.shape}, LUAD={len(luad_ids) if 'luad_ids' in locals() else 100} "
              f"LUSC={len(lusc_ids) if 'lusc_ids' in locals() else 100}")
        return df

    fp = Path(features_csv)
    assert fp.exists(), f"features CSV 不存在: {fp}"
    df = pd.read_csv(fp)
    assert "case_id" in df.columns, "features CSV 缺少 case_id 列"
    print(f"[features] {fp.name}: {df.shape}")
    return df


def build_morphology_matrix(feat_df: pd.DataFrame,
                             omit_cols: list[str] | None = None) -> np.ndarray:
    """从 features DataFrame 抽出形态特征矩阵 (只保留数值列, 自动排除字符串列)。"""
    omit = set(omit_cols or ["case_id", "cohort", "_cohort"])
    # 额外排除非数值列 (has_*, in_tcia_*, 等 omics 对齐标志)
    feat_cols = []
    for c in feat_df.columns:
        if c in omit: continue
        # 只保留数值类型
        if pd.api.types.is_numeric_dtype(feat_df[c]):
            feat_cols.append(c)
    if not feat_cols:
        raise ValueError(f"无有效数值特征列! omit={omit}, cols={feat_df.columns.tolist()[:10]}")
    return feat_df[feat_cols].values.astype(float), feat_cols


# ==============================================================
# 2. 协变量层
# ==============================================================

def build_covariates(clinical_df: pd.DataFrame,
                     case_ids: list[str],
                     cov_names: list[str] | None = None) -> pd.DataFrame:
    """
    从 clinical DataFrame 抽协变量: age, sex, stage, smoking_pack_years。
    返回 (n_cases, n_covs) DataFrame, 行对齐 case_ids (不在 clinical 的 case_id 对应行留 NaN)。
    """
    cov_names = cov_names or ["age", "sex", "stage", "pack_years"]
    if clinical_df.empty:
        return pd.DataFrame(index=case_ids)

    c = clinical_df.copy()

    # case_id 对齐
    if "case_id" not in c.columns and "patient_id" in c.columns:
        c["case_id"] = c["patient_id"].astype(str)
    c = c.set_index("case_id")

    out = pd.DataFrame(index=case_ids)
    common = [cid for cid in case_ids if cid in c.index]

    if not common:
        print(f"[covariates] clinical 与 features case_id 交集为 0, 返回空协变量")
        return out

    print(f"[covariates] clinical 覆盖 {len(common)}/{len(case_ids)} case_id")

    # age
    for col_cand in ["age", "Age"]:
        if col_cand in c.columns:
            out["age"] = np.nan
            out.loc[common, "age"] = c.loc[common, col_cand].astype(float)
            break

    # sex
    for col_cand in ["sex", "Sex"]:
        if col_cand in c.columns:
            out["sex"] = np.nan
            out.loc[common, "sex"] = c.loc[common, col_cand].astype(str).str.upper().map(
                {"MALE": 1, "FEMALE": 0, "M": 1, "F": 0}
            ).astype(float)
            break

    # stage (肿瘤分期)
    for col_cand in ["tumor_stage_pathological", "pathologic_staging_primary_tumor_pt",
                     "pathologic_staging_regional_lymph_nodes_pn"]:
        if col_cand in c.columns:
            col_key = "stage_t" if "pt" in col_cand.lower() else "stage_path"
            out[col_key] = np.nan
            out.loc[common, col_key] = c.loc[common, col_cand].astype(str).str.extract(r"(\d)").astype(float)
            break

    # smoking
    for col_cand in ["number_of_pack_years_smoked", "pack_years_smoked", "pack_years"]:
        if col_cand in c.columns:
            out["pack"] = np.nan
            out.loc[common, "pack"] = c.loc[common, col_cand].astype(float)
            break

    print(f"[covariates] 可用协变量: {list(out.columns)}, 非空率: "
          f"{out.notna().mean().round(3).to_dict()}")
    return out


def residualize(Y: np.ndarray,
                cov: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    对 (n, p) 矩阵 Y 的每一列做协变量线性回归, 返回残差 + r² 解释量。
    自动 dropna (对每列独立); 自动丢弃全 NaN 的协变量列。
    """
    if cov is None or (isinstance(cov, np.ndarray) and cov.size == 0) or \
       (isinstance(cov, pd.DataFrame) and cov.empty):
        return Y.copy(), np.zeros(Y.shape[1])

    cov_ = cov.values if isinstance(cov, pd.DataFrame) else cov
    # 自动丢弃全 NaN 列 (避免它们导致 mask 全 False)
    valid_cols = [j for j in range(cov_.shape[1]) if not np.all(np.isnan(cov_[:, j]))]
    if not valid_cols:
        return Y.copy(), np.zeros(Y.shape[1])
    cov_ = cov_[:, valid_cols]
    # 丢弃只有一个唯一值的列 (常数列无法拟合)
    valid_cols2 = [j for j in range(cov_.shape[1]) if len(np.unique(cov_[:, j][~np.isnan(cov_[:, j])])) > 1]
    if not valid_cols2:
        return Y.copy(), np.zeros(Y.shape[1])
    cov_ = cov_[:, valid_cols2]

    r2s = np.zeros(Y.shape[1])
    out = Y.copy().astype(float)

    for j in range(Y.shape[1]):
        col_mask = ~np.isnan(Y[:, j]) & ~np.isnan(cov_).any(axis=1)
        if col_mask.sum() < 10:
            continue
        lr = LinearRegression()
        lr.fit(cov_[col_mask], Y[col_mask, j])
        out[col_mask, j] = Y[col_mask, j] - lr.predict(cov_[col_mask])
        r2s[j] = lr.score(cov_[col_mask], Y[col_mask, j])

    return out, r2s


# ==============================================================
# 3. 回归引擎
# ==============================================================

def residualize_fit_transform(Y_tr: np.ndarray, Y_va: np.ndarray,
                              cov_tr: np.ndarray, cov_va: np.ndarray):
    """
    fold 内残差化 (消除数据泄露):
      仅在训练集上拟合协变量线性回归, 再对训练/验证集分别取残差。
    返回 (X_tr_resid, X_va_resid)。协变量含 NaN 的样本在每列独立剔除。
    """
    models = {}
    for j in range(Y_tr.shape[1]):
        fit_mask = ~np.isnan(Y_tr[:, j]) & ~np.isnan(cov_tr).any(axis=1)
        if fit_mask.sum() < 10:
            continue
        lr = LinearRegression()
        lr.fit(cov_tr[fit_mask], Y_tr[fit_mask, j])
        models[j] = lr

    def _transform(Y, cov_pred):
        out = Y.copy().astype(float)
        for j, lr in models.items():
            pred_mask = ~np.isnan(Y[:, j]) & ~np.isnan(cov_pred).any(axis=1)
            out[pred_mask, j] = Y[pred_mask, j] - lr.predict(cov_pred[pred_mask])
        return out

    return _transform(Y_tr, cov_tr), _transform(Y_va, cov_va)


def evaluate_enet_nested(X: np.ndarray, y: np.ndarray,
                         n_splits_outer: int = N_SPLITS_OUTER,
                         n_splits_inner: int = N_SPLITS_INNER,
                         n_seeds: int = N_SEEDS,
                         alphas: list[float] | None = None,
                         cov: np.ndarray | None = None,
                         pca_components: int | None = None) -> dict:
    """
    ElasticNet 嵌套 CV:
      外层 5-fold (评估泛化 Pearson / R² / MAE)
      内层 5-fold × 候选 alpha (选最优)
      × n_seeds 次, 最后平均

    防泄露要点 (审稿人重点检查项):
      - 协变量残差化: 只在外层 fold 训练集上拟合残差模型, 再 transform 训练/验证集
      - 标准化: StandardScaler 也只在外层 fold 训练集上 fit
      - pca_components: 若指定, PCA 同样只在每个外层 fold 训练集上 fit (用于高维嵌入基线)

    返回:
      {"pearson_mean", "pearson_std", "r2_mean", "r2_std",
       "mae_mean", "best_alpha", "n_valid"}
    """
    # 精简超参网格: 3 alpha × 1 l1_ratio, 兼顾速度与效果
    alphas = alphas or [0.01, 0.1, 1.0]
    l1_ratios = [0.5]  # elasticnet 混合比
    seeds = list(range(SEED, SEED + n_seeds))

    all_pearson, all_r2, all_mae = [], [], []
    best_alphas = []
    n_valid = 0

    for seed in seeds:
        outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
        for tr_idx, va_idx in outer.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # ---- fold 内协变量残差化 (防泄露) ----
            if cov is not None and cov.size > 0:
                X_tr, X_va = residualize_fit_transform(
                    X_tr, X_va, cov[tr_idx], cov[va_idx])

            # ---- fold 内标准化 (防泄露) ----
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(np.nan_to_num(X_tr, nan=0))
            X_va_s = scaler.transform(np.nan_to_num(X_va, nan=0))

            # ---- fold 内 PCA (防泄露; 高维嵌入基线用) ----
            if pca_components is not None:
                from sklearn.decomposition import PCA
                n_comp = min(pca_components, X_tr_s.shape[1],
                             max(2, X_tr_s.shape[0] - 1))  # 行数/列数都要够
                pca = PCA(n_components=n_comp, random_state=seed)
                X_tr_s = pca.fit_transform(X_tr_s)
                X_va_s = pca.transform(X_va_s)

            inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=seed + 1)
            best_score = -np.inf
            best_params = None

            for alpha in alphas:
                for l1 in l1_ratios:
                    scores = []
                    for i_tr, i_va in inner.split(X_tr_s):
                        m = ElasticNet(alpha=alpha, l1_ratio=l1,
                                       max_iter=5000, random_state=seed, warm_start=False)
                        m.fit(X_tr_s[i_tr], y_tr[i_tr])
                        y_hat = m.predict(X_tr_s[i_va])
                        scores.append(r2_score(y_tr[i_va], y_hat))
                    mean_score = np.mean(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = (alpha, l1)

            alpha_opt, l1_opt = best_params
            m = ElasticNet(alpha=alpha_opt, l1_ratio=l1_opt,
                           max_iter=5000, random_state=seed)
            m.fit(X_tr_s, y_tr)
            y_pred = m.predict(X_va_s)

            if np.std(y_pred) < 1e-8:  # 退化: 预测常数
                continue
            r, _ = pearsonr(y_va, y_pred)
            all_pearson.append(r)
            all_r2.append(r2_score(y_va, y_pred))
            all_mae.append(mean_absolute_error(y_va, y_pred))
            best_alphas.append(alpha_opt)
            n_valid += 1

    if not all_pearson:
        return {"pearson_mean": np.nan, "pearson_std": np.nan,
                "r2_mean": np.nan, "r2_std": np.nan,
                "mae_mean": np.nan, "best_alpha": np.nan, "n_valid": 0}

    return {
        "pearson_mean": float(np.mean(all_pearson)),
        "pearson_std":  float(np.std(all_pearson)),
        "r2_mean":      float(np.mean(all_r2)),
        "r2_std":       float(np.std(all_r2)),
        "mae_mean":     float(np.mean(all_mae)),
        "best_alpha":   float(np.median(best_alphas)),
        "n_valid":      n_valid,
    }


def bootstrap_ci_regression(X: np.ndarray, y: np.ndarray,
                             n_bootstrap: int = BOOTSTRAP_N,
                             seed: int = SEED) -> tuple[float, float]:
    """fit 在完整 X 上, 对 fit 出来的 y_pred 做 patient-level bootstrap Pearson CI。"""
    rng = np.random.RandomState(seed)
    scaler = StandardScaler()
    try:
        m = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=seed)
        m.fit(scaler.fit_transform(X), y)
        p_full = m.predict(scaler.transform(X))
        r_full, _ = pearsonr(y, p_full)
    except Exception:
        return (np.nan, np.nan)

    n = len(y)
    rs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if np.std(y[idx]) < 1e-8:
            continue
        try:
            r, _ = pearsonr(y[idx], p_full[idx])
            rs.append(r)
        except Exception:
            continue

    if len(rs) < 100:
        return (np.nan, np.nan)
    return (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))


def partial_correlation(X: np.ndarray, y: np.ndarray,
                         cov: pd.DataFrame | None = None,
                         top_k_features: int = 3) -> float:
    """
    用 ElasticNet fit X→y, 取 top_k_features 系数的特征列, 分别算偏相关, 取最大绝对值返回。
    若 cov 非空 → 先残差化再 pearson。
    """
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    cov_arr = cov.values if isinstance(cov, pd.DataFrame) else cov
    if cov_arr is not None and cov_arr.ndim == 2 and cov_arr.shape[1] > 0:
        mask &= ~np.isnan(cov_arr).any(axis=1)
    if mask.sum() < 10:
        return np.nan

    X_m = X[mask]
    y_m = y[mask]
    cov_m = cov_arr[mask] if cov_arr is not None and cov_arr.ndim == 2 and cov_arr.shape[1] > 0 else None

    # 1) ElasticNet fit → 取系数 top features
    try:
        m = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=SEED)
        scaler = StandardScaler()
        m.fit(scaler.fit_transform(X_m), y_m)
        top_idx = np.argsort(np.abs(m.coef_))[::-1][:top_k_features]
    except Exception:
        top_idx = [0]  # fallback 只用第一列

    # 2) 对 top 特征分别算偏相关 (cov 非空时先残差化)
    results = []
    for j in top_idx:
        xv = X_m[:, j]
        if cov_m is not None:
            lr_x = LinearRegression().fit(cov_m, xv)
            lr_y = LinearRegression().fit(cov_m, y_m)
            xv = xv - lr_x.predict(cov_m)
            yv = y_m - lr_y.predict(cov_m)
        try:
            r, _ = pearsonr(xv, yv)
            results.append(abs(r))
        except Exception:
            continue

    return float(max(results)) if results else np.nan


# ==============================================================
# 4. BH-FDR 校正
# ==============================================================

def bh_fdr(pvalues: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正。返回 q-values。"""
    p = np.array(pvalues, dtype=float)
    order = np.argsort(p)
    p_sorted = p[order]
    n = len(p)
    q = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        if i == n - 1:
            q[i] = min(p_sorted[i] * n / rank, 1.0)
        else:
            q[i] = min(p_sorted[i] * n / rank, q[i + 1])
    result = np.empty(n)
    result[order] = q
    return result


# ==============================================================
# 5. 协变量校正检验
# ==============================================================

def permutation_test(X: np.ndarray, y: np.ndarray,
                      seed: int = SEED,
                      n_perm: int = 1000,
                      model_fn=None) -> float:
    """置换检验 p-value (把 y 打乱后再跑回归, 看 r² 分位)。"""
    rng = np.random.RandomState(seed)
    if model_fn is None:
        # 用简单 ElasticNet 跑一次 baseline
        def model_fn(X_tr, y_tr, X_va):
            m = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=seed)
            m.fit(X_tr, y_tr)
            return m.predict(X_va)

    outer = KFold(n_splits=5, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(np.nan_to_num(X, nan=0))
    obs_r2s = []
    null_r2s = []

    for tr_idx, va_idx in outer.split(X_s):
        y_pred_obs = model_fn(X_s[tr_idx], y[tr_idx], X_s[va_idx])
        obs_r2s.append(r2_score(y[va_idx], y_pred_obs))

    obs_mean = np.mean(obs_r2s)
    for _ in range(n_perm):
        y_shuf = rng.permutation(y)
        null_r2s = []
        for tr_idx, va_idx in outer.split(X_s):
            y_pred_null = model_fn(X_s[tr_idx], y_shuf[tr_idx], X_s[va_idx])
            null_r2s.append(r2_score(y_shuf[va_idx], y_pred_null))
        # 保留 1 次置换的 mean (避免 null 分布过窄)
        null_r2s.append(np.mean(null_r2s))

    # p = 1 - ECDF(obs)  null 分布下 obs 有多极端
    p = float(1 - (np.mean(null_r2s) <= obs_mean))
    # 更保守: null_mean > obs 才记 sig
    p = float(np.mean([1.0] + [0.0 for nr in null_r2s if nr > obs_mean]))
    return p


# ==============================================================
# 6. 主循环
# ==============================================================

def run_single_target(X: np.ndarray, y: np.ndarray,
                       target_name: str,
                       cov: pd.DataFrame | None = None,
                       seed: int = SEED,
                       do_permutation: bool = True,
                       pca_components: int | None = None) -> dict:
    """对单个 target 跑完整评估流水线, 返回结果 dict。"""
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    if mask.sum() < 20 or np.std(y[mask]) < 1e-8:
        return {
            "target": target_name, "n_valid": int(mask.sum()),
            "pearson": np.nan, "pearson_std": np.nan, "r2": np.nan,
            "r2_std": np.nan, "mae": np.nan, "best_alpha": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "partial_r": np.nan,
            "permutation_p": np.nan,
        }

    X_ok = X[mask]
    y_ok = y[mask]
    cov_ok = cov[mask] if cov is not None and cov.shape[0] == X.shape[0] else None

    # 嵌套 CV ElasticNet (fold 内残差化 + 标准化 + 可选 fold 内 PCA, 防泄露)
    enet = evaluate_enet_nested(X_ok, y_ok, cov=cov_ok, pca_components=pca_components)
    # bootstrap CI
    ci_lo, ci_hi = bootstrap_ci_regression(X_ok, y_ok, seed=seed)
    # 偏相关
    pr = partial_correlation(X_ok, y_ok, cov_ok)
    # permutation p-value (或快速 pearson p-value)
    if do_permutation:
        p_perm = permutation_test(X_ok, y_ok, seed=seed, n_perm=500)
    else:
        # 快速方案: 用 CV 外 pearson 均值做 t 检验 (不依赖训练集过拟合)
        r = enet["pearson_mean"]
        n = int(mask.sum())
        if n > 2 and not np.isnan(r) and abs(r) < 1.0:
            from scipy.stats import t as t_dist
            t_stat = r * np.sqrt((n - 2) / (1 - r**2))
            p_perm = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 2))
        else:
            p_perm = np.nan

    return {
        "target":       target_name,
        "n_valid":      int(mask.sum()),
        "pearson":      enet["pearson_mean"],
        "pearson_std":  enet["pearson_std"],
        "r2":           enet["r2_mean"],
        "r2_std":       enet["r2_std"],
        "mae":          enet["mae_mean"],
        "best_alpha":   enet["best_alpha"],
        "ci_low":       ci_lo,
        "ci_high":      ci_hi,
        "partial_r":    pr,
        "permutation_p": p_perm,
    }


def run_modality(X: np.ndarray, omics_df: pd.DataFrame,
                  cov: pd.DataFrame | None = None,
                  modality_name: str = "proteomics",
                  top_k_variance: int = 2000,
                  save_every: int = 50,
                  out_csv: str | None = None,
                  seed: int = SEED,
                  n_seeds: int = N_SEEDS,
                  feature_names: list[str] | None = None,
                  save_importance: str | None = None,
                  do_permutation: bool = True,
                  pca_components: int | None = None) -> pd.DataFrame:
    """
    对一个组学模态 (proteomics/phospho/RNA) 批量跑所有 target。
    - 先筛高方差 target (top 25% 或 top_k 上限)
    - 每个 target 跑嵌套 CV + BH-FDR
    - 可选: 保存 top hits 的特征重要性 (niche abundance / Ripley K 系数)
    """
    # 对齐样本
    id_col = "case_id" if "case_id" in omics_df.columns else (
        "patient_id" if "patient_id" in omics_df.columns else None)
    assert id_col, "omics DataFrame 缺少 ID 列"
    ids = omics_df[id_col].astype(str).tolist()
    # 排除所有非数值列
    non_target = {"patient_id", "case_id", "_cohort", "cohort"}
    target_cols = [c for c in omics_df.columns if c not in non_target]
    omics_mat = omics_df[target_cols].apply(pd.to_numeric, errors="coerce").values

    # 方差筛选 (跳过全 NaN 列, 否则 nanvar=nan 会被 argsort 排到最前)
    valid_cols = np.array([not np.all(np.isnan(omics_mat[:, j])) for j in range(omics_mat.shape[1])])
    target_cols = [c for c, v in zip(target_cols, valid_cols) if v]
    omics_mat = omics_mat[:, valid_cols]
    variances = np.nanvar(omics_mat, axis=0)
    order = np.argsort(variances)[::-1]
    keep_n = min(top_k_variance, len(target_cols))
    selected = order[:keep_n]
    print(f"[{modality_name}] target 总数 {len(target_cols)} (过滤全空后), 高方差筛选 {keep_n}")

    results = []
    start = time.time()
    for i, idx in enumerate(selected):
        target_name = target_cols[idx]
        y = omics_mat[:, idx]
        r = run_single_target(X, y, target_name, cov=cov, seed=seed,
                              do_permutation=do_permutation,
                              pca_components=pca_components)
        r["modality"] = modality_name
        results.append(r)
        if (i + 1) % save_every == 0 or i == keep_n - 1:
            t = (time.time() - start) / (i + 1) * (keep_n - i - 1) if i > 0 else 0
            print(f"  [{i+1}/{keep_n}] pearson={r['pearson']:.3f}  "
                  f"n={r['n_valid']}  预估剩余 {t/60:.1f} min", flush=True)

    df = pd.DataFrame(results)
    # BH-FDR
    if "permutation_p" in df.columns:
        qs = bh_fdr(df["permutation_p"].fillna(1.0).values)
        df["q_value"] = qs

    sig = df[df["q_value"] < 0.05] if "q_value" in df.columns else pd.DataFrame()
    print(f"[{modality_name}] BH-FDR q<0.05 显著: {len(sig)}/{len(df)}")
    if not sig.empty:
        print("  Top 5:")
        for _, r in sig.nlargest(5, "pearson").iterrows():
            print(f"    {r['target']:40s}  pearson={r['pearson']:.3f}  q={r['q_value']:.4f}  n={r['n_valid']}")

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[{modality_name}] 结果 → {out_path}")

    return df


# ==============================================================
# 7. 跨 target 特征重要性聚合
# ==============================================================

def aggregate_feature_importance(X: np.ndarray, omics_df: pd.DataFrame,
                                  target_cols: list[str],
                                  top_significant_q: float = 0.05,
                                  modality_name: str = "proteomics",
                                  seed: int = SEED) -> pd.DataFrame:
    """
    对每个显著 target 重新 fit ElasticNet 拿系数, 跨 target 取绝对值均值
    → 得到每个形态特征在预测"哪些蛋白/位点/基因"上的贡献度排序。
    """
    import sys
    target_ids = [c for c in target_cols if c != "patient_id" and c != "_cohort"]
    id_col = "patient_id" if "patient_id" in omics_df.columns else "case_id"
    omics_mat = omics_df[target_ids].apply(pd.to_numeric, errors="coerce").values

    all_coefs = []
    scaler = StandardScaler()
    X_s = scaler.fit_transform(np.nan_to_num(X, nan=0))

    for i, target_name in enumerate(target_ids):
        y = omics_mat[:, i]
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        if mask.sum() < 30:
            continue
        try:
            m = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=seed)
            m.fit(X_s[mask], y[mask])
            if np.abs(m.coef_).sum() > 1e-8:
                all_coefs.append(np.abs(m.coef_))
        except Exception:
            continue

    if not all_coefs:
        return pd.DataFrame(columns=["feature", "mean_abs_coef", "support_count", "modality"])

    coef_stack = np.vstack(all_coefs)
    mean_abs = coef_stack.mean(axis=0)
    support = (coef_stack > 0).sum(axis=0)  # 被非零系数选中的 target 数

    feat_names = [f"f{i}" for i in range(X.shape[1])]
    out = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_coef": mean_abs,
        "support_count": support,
        "modality": modality_name,
    }).sort_values("mean_abs_coef", ascending=False)
    return out


# ==============================================================
# 8. Survival 附加
# ==============================================================

def run_os_analysis(X: np.ndarray, clinical_df: pd.DataFrame,
                    case_ids: list[str],
                    cov: pd.DataFrame | None = None,
                    seed: int = SEED) -> pd.DataFrame:
    """
    形态特征 + 协变量 → OS Cox (elasticnet) + C-index 5-fold CV。
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        print("[OS] lifelines 未安装, 跳过 (pip install lifelines)")
        return pd.DataFrame()

    if clinical_df.empty:
        return pd.DataFrame()

    c = clinical_df.copy()
    if "case_id" not in c.columns and "patient_id" in c.columns:
        c["case_id"] = c["patient_id"].astype(str)

    # 时间 + event
    t_col, e_col = None, None
    for tc in ["Overall survival, days", "overall_survival_days", "OS_days"]:
        if tc in c.columns: t_col = tc; break
    for ec in ["Survival status (1, dead; 0, alive)", "survival_status", "OS_event"]:
        if ec in c.columns: e_col = ec; break

    if not t_col or not e_col:
        print("[OS] 找不到生存时间/状态列, 跳过")
        return pd.DataFrame()

    c = c[c["case_id"].isin(case_ids)].copy()
    c["time"] = c[t_col].astype(float)
    c["event"] = c[e_col].astype(float)
    c = c[c["time"].notna() & c["event"].notna() & (c["time"] > 0)]

    if len(c) < 50:
        print(f"[OS] 有效样本仅 {len(c)}, 跳过")
        return pd.DataFrame()

    id_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    valid = [cid for cid in c["case_id"] if cid in id_to_idx]
    X_sub = np.array([X[id_to_idx[cid]] for cid in valid])
    time = c.set_index("case_id").loc[valid, "time"].values
    event = c.set_index("case_id").loc[valid, "event"].values

    # PCA 降维 (236 维 > 194 样本会导致矩阵奇异/共线性)
    from sklearn.decomposition import PCA
    n_comp = min(50, X_sub.shape[1], X_sub.shape[0] - 1)
    print(f"[OS] PCA 目标 {X_sub.shape[1]} → {n_comp} 维 (在每折训练集上拟合, 防泄露)")

    skf = StratifiedKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=seed)
    cidxs = []

    for tr_idx, va_idx in skf.split(np.zeros(len(time)), event):
        # ---- fold 内标准化 + PCA (防泄露) ----
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(np.nan_to_num(X_sub[tr_idx], nan=0))
        X_va_s = scaler.transform(np.nan_to_num(X_sub[va_idx], nan=0))
        pca = PCA(n_components=n_comp, random_state=seed)
        X_tr_pca = pca.fit_transform(X_tr_s)
        X_va_pca = pca.transform(X_va_s)

        df_train = pd.DataFrame(X_tr_pca, columns=[f"pc{i}" for i in range(n_comp)])
        df_train["time"] = time[tr_idx]
        df_train["event"] = event[tr_idx]
        df_val = pd.DataFrame(X_va_pca, columns=[f"pc{i}" for i in range(n_comp)])
        df_val["time"] = time[va_idx]
        df_val["event"] = event[va_idx]
        try:
            cph = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
            cph.fit(df_train, duration_col="time", event_col="event", show_progress=False)
            pred = cph.predict_partial_hazard(df_val.drop(columns=["time", "event"]))
            pred_vals = pred.values.ravel() if hasattr(pred, 'values') else np.asarray(pred).ravel()
            cidx = concordance_index(time[va_idx], -pred_vals, event[va_idx])
            cidxs.append(cidx)
        except Exception as e:
            print(f"  [OS fold] error: {e}")
            continue

    df = pd.DataFrame([{
        "analysis": "OS_CoxPH",
        "n": len(time),
        "cindex_mean": float(np.mean(cidxs)) if cidxs else np.nan,
        "cindex_std":  float(np.std(cidxs)) if cidxs else np.nan,
        "event_rate":  float(event.mean()),
    }])
    return df


# ==============================================================
# 9. 主入口
# ==============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True,
                    help="形态特征 CSV (run_cptac.py 输出)")
    ap.add_argument("--omics-dir", type=str, default=None,
                    help="CPTAC omics 目录 (默认 data/raw/CPTAC/omics)")
    ap.add_argument("--mock", action="store_true",
                    help="用合成特征矩阵跑 sanity check (CPU 友好, 不跑真实嵌入)")
    ap.add_argument("--cohort", choices=["ALL", "LUAD", "LUSC"], default="ALL",
                    help="分层分析 (默认 ALL)")
    ap.add_argument("--targets-top-k", type=int, default=2000,
                    help="每个模态跑 top k 高方差 target (默认 2000)")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="CV 随机种子数")
    ap.add_argument("--modalities", type=str, nargs="+",
                    default=["proteomics", "phosphoproteomics", "transcriptomics"],
                    help="只加载指定模态 (默认全部), 如 --modalities proteomics")
    ap.add_argument("--no-permutation", action="store_true",
                    help="跳过置换检验, 用 scipy pearsonr p-value 做 FDR (快 10x, smoke test 推荐)")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    print(f"\n{'='*70}\n[downstream_omics] features={args.features} cohort={args.cohort} seed={SEED}\n{'='*70}")

    # 1. 先加载 omics (mock 模式也需要 roster 拿真实 case_id)
    omics = load_cptac_omics(args.omics_dir, modalities=args.modalities)
    omics = unify_patient_ids(omics)

    # 2. 加载/生成 features
    feat_df = load_morphology_features(args.features, mock=args.mock,
                                        real_roster=omics.get("roster"))

    # 3. 按 cohort 筛 (ALL 保留全部, 其他只留对应癌种)
    cohort_col = "_cohort" if "_cohort" in feat_df.columns else "cohort"
    if args.cohort != "ALL" and cohort_col in feat_df.columns:
        feat_df = feat_df[feat_df[cohort_col].astype(str).str.upper() == args.cohort.upper()].copy()
        print(f"[cohort filter] {args.cohort}: {len(feat_df)} cases")

    # 3. features × omics × clinical merge
    case_ids = feat_df["case_id"].astype(str).tolist()
    X_all, feat_names = build_morphology_matrix(feat_df)
    print(f"[merge] features n={len(case_ids)}, omics case_id 类型多样, 各表独立 inner join")

    # 4. 协变量
    cov = None
    if "clinical" in omics:
        cov = build_covariates(omics["clinical"], case_ids)

    # 注: 协变量残差化不再在全集上做 —— 已在 evaluate_enet_nested 外层 fold 内完成 (防泄露)

    # 6. 逐模态跑
    out_dir = Path(args.out_dir) if args.out_dir else \
        (config.RESULTS / "omics" if hasattr(config, "RESULTS")
         else Path(__file__).parent.parent / "results" / "omics")
    out_dir.mkdir(parents=True, exist_ok=True)

    # case_id -> 特征矩阵行号 的映射 (X_all 与 feat_df 顺序一致)
    idx_of = {cid: i for i, cid in enumerate(case_ids)}

    modality_results = {}
    for mod in ["proteomics", "phosphoproteomics", "transcriptomics"]:
        if mod not in omics:
            continue
        om_df = omics[mod].copy()
        # 对齐样本
        om_df = om_df[om_df["case_id"].isin(case_ids)].reset_index(drop=True)
        n_match = len(om_df)
        if n_match < 50:
            print(f"[{mod}] 与 features 交集仅 {n_match}, 跳过")
            continue
        print(f"\n{'─'*70}\n[{mod}] 交集 {n_match} / {len(case_ids)}\n{'─'*70}")

        # 对齐协变量 + 特征矩阵: 全部按 om_df 的 case_id 顺序重排
        om_ids = om_df["case_id"].astype(str).tolist()
        row_idx = [idx_of[c] for c in om_ids]
        X_mod = X_all[row_idx]                       # 原始特征 (未残差化, fold 内做)
        cov_mod = cov.loc[om_ids].values if cov is not None else None  # 与 X_mod 同序

        r_df = run_modality(
            X=X_mod,
            omics_df=om_df,
            cov=cov_mod,
            modality_name=mod,
            top_k_variance=args.targets_top_k,
            out_csv=str(out_dir / f"{args.cohort}_{mod}_results.csv"),
            n_seeds=args.seeds,
            feature_names=feat_names,
            do_permutation=not args.no_permutation,
        )
        modality_results[mod] = r_df

        # 特征重要性
        imp = aggregate_feature_importance(X_mod, om_df, target_cols=r_df.get("target", []),
                                            modality_name=mod, seed=SEED)
        if not imp.empty:
            imp.to_csv(out_dir / f"{args.cohort}_{mod}_importance.csv", index=False)
            print(f"  特征重要性 top 5: {imp.head(5)['feature'].tolist()}")

    # 7. Survival
    if "clinical" in omics:
        os_df = run_os_analysis(X_all, omics["clinical"], case_ids, cov=cov)
        if not os_df.empty:
            os_df.to_csv(out_dir / f"{args.cohort}_os_cox.csv", index=False)
            print(f"\n[OS Cox] C-index = {os_df.iloc[0]['cindex_mean']:.3f} ± "
                  f"{os_df.iloc[0]['cindex_std']:.3f}")

    # 8. 汇总
    all_df = []
    for mod, df in modality_results.items():
        all_df.append(df)
    if all_df:
        summary = pd.concat(all_df, ignore_index=True)
        sig = summary[summary.get("q_value", pd.Series([1.0]*len(summary))) < 0.05]
        print(f"\n{'='*70}")
        print(f"[最终汇总] 模态 {len(modality_results)}, 所有 target {len(summary)}, BH-FDR q<0.05 显著 {len(sig)}")
        if not sig.empty:
            print("[Top hits by modality:")
            for mod, g in sig.groupby("modality"):
                top = g.nlargest(3, "pearson")
                for _, r in top.iterrows():
                    print(f"  [{mod:18s}] {r['target']:40s} r={r['pearson']:.3f} "
                          f"q={r.get('q_value', 1):.4f} n={r['n_valid']}")
        summary.to_csv(out_dir / f"{args.cohort}_all_modalities_summary.csv", index=False)
        print(f"[summary] → {out_dir / f'{args.cohort}_all_modalities_summary.csv'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
