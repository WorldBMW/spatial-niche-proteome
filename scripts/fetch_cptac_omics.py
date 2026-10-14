"""
fetch_cptac_omics.py — 用 PayneLab cptac 包拉取 CPTAC-LUAD/LSCC 多组学表并导出 CSV，
                       同时生成 C3L-/C3N- patient_id 对齐名册 (roster)。

数据源: https://paynelab.github.io/cptac/  (底层为 PDC / S3, 大陆需代理)

用法 (PowerShell):
  $env:HTTPS_PROXY='http://127.0.0.1:7890'; $env:HTTP_PROXY='http://127.0.0.1:7890'
  python scripts/fetch_cptac_omics.py --cohort luad --tables clinical   # 小流量测通
  python scripts/fetch_cptac_omics.py --cohort both                     # 全量拉取

输出 (data/raw/CPTAC/omics/):
  {luad|lusc}_clinical.csv            临床信息 (含生存、分期等)
  {luad|lusc}_proteomics.csv          全局蛋白组 (log-ratio, 行为患者)
  {luad|lusc}_phosphoproteomics.csv   磷酸化蛋白组 (gene|site 列)
  {luad|lusc}_transcriptomics.csv     RNA-seq
  {luad|lusc}_somatic_mutation.csv    体细胞突变 (长表)
  patient_roster.csv                  患者对齐表: 各表有无 + 是否在 TCIA 影像清单
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import cptac

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "CPTAC" / "omics"
TCIA_SERIES = ROOT / "data" / "raw" / "CPTAC" / "cptac_tcia_all_series.csv"

COHORTS = {"luad": cptac.Luad, "lusc": cptac.Lscc}  # PayneLab 类名为 Lscc
ALL_TABLES = ["clinical", "proteomics", "phosphoproteomics",
              "transcriptomics", "somatic_mutation"]

# cptac 1.5.14 不传 source 会触发 generator len() 的上游 bug, 必须显式指定。
# 统一选用 CPTAC 原始论文的标准 source。
TABLE_SOURCE = {
    "clinical": "mssm",
    "proteomics": "umich",
    "phosphoproteomics": "umich",
    "transcriptomics": "bcm",
    "somatic_mutation": "washu",
}


def flatten_columns(df: pd.DataFrame, keep_site: bool = False) -> pd.DataFrame:
    """把 MultiIndex 列压成单列字符串。
    keep_site=True (磷酸化/突变): 保留全部层级 gene|site...
    keep_site=False (蛋白/RNA): 用第一层级 gene symbol, 重复加序号。
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    names, seen = [], {}
    for tup in df.columns:
        parts = [str(x) for x in tup if str(x) not in ("nan", "None", "")]
        if keep_site:
            name = "|".join(dict.fromkeys(parts)) if parts else "NA"
        else:
            name = parts[0] if parts else "NA"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}__{seen[name]}"
        names.append(name)
    out = df.copy()
    out.columns = names
    return out


def get_table(ds, table: str) -> pd.DataFrame:
    src = TABLE_SOURCE[table]
    if table == "clinical":
        return ds.get_clinical(source=src)
    if table == "proteomics":
        return flatten_columns(ds.get_proteomics(source=src))
    if table == "phosphoproteomics":
        return flatten_columns(ds.get_phosphoproteomics(source=src), keep_site=True)
    if table == "transcriptomics":
        return flatten_columns(ds.get_transcriptomics(source=src))
    if table == "somatic_mutation":
        # 优先 binary (基因×患者 0/1 矩阵); 该源没有时回退长表 (每行一个突变事件)
        try:
            return flatten_columns(ds.get_somatic_mutation_binary(source=src), keep_site=True)
        except Exception:
            return ds.get_somatic_mutation(source=src)
    raise ValueError(table)


def export(ds, cohort: str, tables: list[str], force: bool) -> dict:
    available = {}
    for t in tables:
        fp = OUT / f"{cohort}_{t}.csv"
        if fp.exists() and not force:
            print(f"  [skip] {t}: {fp.name} 已存在 (--force 重拉)")
            ids = pd.read_csv(fp, usecols=["patient_id"])["patient_id"].astype(str)
            # 只统计肿瘤样本: cptac 中癌旁样本 id 以 .N 结尾
            available[t] = set(ids[~ids.str.endswith(".N")])
            continue
        print(f"  [get] {t} ...", flush=True)
        t0 = time.time()
        try:
            df = get_table(ds, t)
        except Exception as e:
            print(f"  [FAIL] {t}: {type(e).__name__}: {e}")
            continue
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "patient_id"})
        df["patient_id"] = df["patient_id"].astype(str)
        # 标注组织类型: 末尾 .N 为癌旁正常组织, 其余为肿瘤
        df.insert(1, "tissue_type",
                  np.where(df["patient_id"].str.endswith(".N"), "normal", "tumor"))
        df.to_csv(fp, index=False)
        n_tum = int((df["tissue_type"] == "tumor").sum())
        print(f"  [ok] {t}: shape={df.shape} (tumor={n_tum}) "
              f"({time.time()-t0:.0f}s) -> {fp.name}")
        available[t] = set(df.loc[df["tissue_type"] == "tumor", "patient_id"])
    return available


def build_roster(cohort: str, available: dict, tcia_ids: set[str]) -> pd.DataFrame:
    all_ids = sorted(set().union(*available.values())) if available else []
    r = pd.DataFrame({"patient_id": all_ids, "cohort": cohort})
    for t in ALL_TABLES:
        r[f"has_{t}"] = r["patient_id"].isin(available.get(t, set())).astype(int)
    r["in_tcia_imaging"] = r["patient_id"].isin(tcia_ids).astype(int)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["luad", "lusc", "both"], default="both")
    ap.add_argument("--tables", nargs="+", default=ALL_TABLES, choices=ALL_TABLES)
    ap.add_argument("--force", action="store_true", help="已存在的 CSV 也重新拉取")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore")

    tcia_ids = set()
    if TCIA_SERIES.exists():
        tcia_ids = set(pd.read_csv(TCIA_SERIES)["PatientID"].astype(str))
        print(f"[roster] TCIA 影像清单患者数: {len(tcia_ids)}")

    cohorts = ["luad", "lusc"] if args.cohort == "both" else [args.cohort]
    rosters = []
    for key in cohorts:
        print(f"\n=== {key.upper()}: 初始化数据集 (首次会下载数据索引) ===", flush=True)
        ds = COHORTS[key]()
        available = export(ds, key, args.tables, args.force)
        rosters.append(build_roster(key, available, tcia_ids))

    if rosters:
        roster = pd.concat(rosters, ignore_index=True)
        rp = OUT / "patient_roster.csv"
        roster.to_csv(rp, index=False)
        print(f"\n[roster] {rp}: {len(roster)} patients")
        for key, grp in roster.groupby("cohort"):
            flags = {t: int(grp[f"has_{t}"].sum()) for t in ALL_TABLES}
            print(f"  {key}: n={len(grp)}, in_tcia={int(grp['in_tcia_imaging'].sum())}, {flags}")


if __name__ == "__main__":
    sys.exit(main())
