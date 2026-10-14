"""
data_loader.py
==============
统一的数据加载 + case_id merge 层。

解决的核心问题：
  - features CSV 里的 case_id (未来是 TCGA-XX-XXXX)
    与 tcga_clinical_full.csv / tcga_os_survival.csv 的 case_id 统一
  - CPTAC 的 C3L-XXXXX 与 CPTAC-LUAD/LUSC 临床 CSV 的 case_id 统一
  - 自动 inner join 取交集，避免 case_id 格式不一致导致 NaN

所有函数返回的 DataFrame 都已清理好 case_id 列。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from case_id import resolve_case_id


# ==============================================================
# TCGA 数据
# ==============================================================

def load_tcga_clinical() -> pd.DataFrame:
    """
    合并 tcga_clinical_full (突变+TMB+FGA+cohort) + tcga_os_survival (OS+AJCC)。
    返回 DataFrame: 每行一个 TCGA 患者，列含所有突变、TMB、FGA、OS、AJCC、cohort。
    """
    clin_path = config.DATA_RAW / "TCGA" / "tcga_clinical_full.csv"
    os_path = config.DATA_RAW / "TCGA" / "tcga_os_survival.csv"

    if not clin_path.exists():
        # fallback: SRC_EXP1 里的旧文件
        clin_path = config.SRC_TCGA_CLIN
    if not os_path.exists():
        os_path = config.DATA_RAW / "TCGA" / "tcga_os_survival.csv"

    clin = pd.read_csv(clin_path)
    os_df = pd.read_csv(os_path) if os_path.exists() else pd.DataFrame()

    # 统一 case_id
    clin["case_id"] = clin["case_id"].apply(resolve_case_id)
    if not os_df.empty:
        os_df["case_id"] = os_df["case_id"].apply(resolve_case_id)

    # 去重（同一 case_id 可能多行）
    clin = clin.drop_duplicates(subset=["case_id"], keep="first")
    if not os_df.empty:
        os_df = os_df.drop_duplicates(subset=["case_id"], keep="first")

    # merge
    if not os_df.empty:
        merged = clin.merge(
            os_df[["case_id", "OS_MONTHS", "OS_STATUS", "AJCC_STAGE"]],
            on="case_id", how="left"
        )
    else:
        merged = clin
        merged["OS_MONTHS"] = np.nan
        merged["OS_STATUS"] = np.nan
        merged["AJCC_STAGE"] = np.nan

    # cohort 列
    if "cohort" not in merged.columns:
        merged["cohort"] = "TCGA"

    print(f"[data_loader] TCGA clinical: {len(merged)} patients, "
          f"{merged['cohort'].value_counts().to_dict()}")
    print(f"  OS non-null: {merged['OS_MONTHS'].notna().sum()}, "
          f"OS_STATUS non-null: {merged['OS_STATUS'].notna().sum()}")
    print(f"  AJCC_STAGE non-null: {merged['AJCC_STAGE'].notna().sum()}")
    return merged


def load_cptac_clinical() -> pd.DataFrame:
    """
    合并 CPTAC-LUAD + CPTAC-LUSC 临床 CSV。
    返回 DataFrame: 每行一个 CPTAC 患者。
    """
    luad_path = config.SRC_CPTAC_LUAD
    lusc_path = config.SRC_CPTAC_LUSC

    if not luad_path.exists():
        # 在 experiment2/data/raw/CPTAC/ 找
        candidates = list((config.DATA_RAW / "CPTAC").glob("*CPTAC*LUAD*.csv"))
        if candidates:
            luad_path = candidates[0]
        else:
            print(f"[data_loader] WARNING: CPTAC-LUAD clinical not found")
            return pd.DataFrame()

    luad = pd.read_csv(luad_path)
    lusc = pd.read_csv(lusc_path) if lusc_path.exists() else pd.DataFrame()

    luad["cohort"] = "CPTAC-LUAD"
    if not lusc.empty:
        lusc["cohort"] = "CPTAC-LUSC"
        df = pd.concat([luad, lusc], ignore_index=True)
    else:
        df = luad

    # 统一 case_id — CPTAC clinical CSV 里是 patientId, 有的是 C3L-XXXXX, 有的是其他格式
    # 重命名为 case_id
    if "patientId" in df.columns:
        df = df.rename(columns={"patientId": "case_id"})
    elif "sampleId" in df.columns:
        df = df.rename(columns={"sampleId": "case_id"})

    df["case_id"] = df["case_id"].astype(str).apply(resolve_case_id)
    df = df.drop_duplicates(subset=["case_id"], keep="first")

    print(f"[data_loader] CPTAC clinical: {len(df)} patients, "
          f"{df['cohort'].value_counts().to_dict()}")
    return df


# ==============================================================
# Features 加载 + merge
# ==============================================================

def load_features(backbone: str) -> pd.DataFrame:
    """
    加载 features CSV。
    注意：如果 CSV 里 case_id 是旧的 TCGA-S0000 / CPTAC-S0000 格式，
    需要手动修复（见 scripts/remap_old_features.py）。
    理想情况是 patch_embed 已用 case_id.resolve_case_id() 重跑过。
    """
    fp = config.features_dir(backbone) / "all_features.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"features CSV not found: {fp}\n"
            f"先跑: python src/pipeline.py --step features --backbone {backbone}"
        )
    df = pd.read_csv(fp)
    df["case_id"] = df["case_id"].astype(str).apply(resolve_case_id)
    print(f"[data_loader] features ({backbone}): {len(df)} cases, "
          f"{len([c for c in df.columns if c != 'case_id'])} features")
    return df


def merge_features_clinical(features_df: pd.DataFrame,
                              clinical_df: pd.DataFrame,
                              label: str = "") -> pd.DataFrame:
    """
    features × clinical 按 case_id inner join。
    返回交集 DataFrame。
    """
    feat_ids = set(features_df["case_id"])
    clin_ids = set(clinical_df["case_id"])
    common = feat_ids & clin_ids

    if not common:
        print(f"[data_loader] {label} WARNING: 0 case_id 交集!")
        print(f"  features 示例: {list(feat_ids)[:3]}")
        print(f"  clinical 示例: {list(clin_ids)[:3]}")
        print(f"  features 总: {len(feat_ids)}, clinical 总: {len(clin_ids)}")
        # 尝试 partial string match (有时 clinical 用 C3L-, features 用 CPTAC-)
        return pd.DataFrame()

    merged = features_df.merge(clinical_df, on="case_id", how="inner")
    print(f"[data_loader] {label} merged: {len(merged)} / "
          f"{len(features_df)} features × {len(clinical_df)} clinical")
    return merged


# ==============================================================
# 突变任务发现
# ==============================================================

def discover_mutation_tasks(clinical_df: pd.DataFrame,
                              min_n: int = 20,
                              min_pos: int = 3) -> list[dict]:
    """
    从 clinical DataFrame 里自动发现可用的二值突变任务。
    
    Returns:
        list[dict], 每个 dict 含:
            col: 原始列名
            task: 任务名 (col_mutation)
            n: 样本量
            pos_rate: 阳性率
    """
    tasks = []
    for col in clinical_df.columns:
        if col in ("case_id", "cohort", "OS_MONTHS", "OS_STATUS", "AJCC_STAGE",
                    "FRACTION_GENOME_ALTERED", "TMB_NONSYNONYMOUS"):
            continue
        series = clinical_df[col].dropna()
        # 必须是 0/1 二值
        unique = set(series.unique())
        if unique <= {0, 1, 0.0, 1.0, True, False}:
            series_int = series.astype(int)
            n = len(series_int)
            pos = int(series_int.sum())
            neg = n - pos
            if n >= min_n and pos >= min_pos and neg >= min_pos:
                tasks.append({
                    "col": col,
                    "task": f"{col}_mutation" if col.upper() not in ("TMB_HIGH", "CIMP_HIGH") else col,
                    "n": n,
                    "pos": pos,
                    "neg": neg,
                    "pos_rate": pos / n,
                })
    # 按 pos_rate 排序，太高或太低的放后面
    tasks.sort(key=lambda x: abs(x["pos_rate"] - 0.5))
    return tasks


# ==============================================================
# Cohort 分离
# ==============================================================

def split_by_cohort(df: pd.DataFrame, cohort_col: str = "cohort") -> dict[str, pd.DataFrame]:
    """按 cohort 列拆分。返回 {cohort_name: subset_df}。"""
    if cohort_col not in df.columns:
        return {"ALL": df}
    return {name: sub for name, sub in df.groupby(cohort_col)}
