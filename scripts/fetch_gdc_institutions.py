"""GDC 机构查询 + cross-institution split — 本地纯 CPU 跑, 生成 train/test split CSV.

背景:
    TCGA-LUAD/LUSC 的 case_id (TCGA-XX-XXXX) = GDC submitter_id
    GDC cases API 可以查 submitter_id → project.project_id (机构 TSS)
    拿到机构后按机构切分 train/test (70%/30% 机构, 保证外部验证独立性)

输出:
    data/raw/TCGA/tcga_clinical_with_institution.csv  (clinical + 机构 + split)
    data/raw/TCGA/institution_summary.csv              (机构样本量分布)

用法:
    python scripts/fetch_gdc_institutions.py
"""
import json, time, urllib.request, urllib.parse, sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"e:\ai_paper\experiment2_fm_time")
CLINICAL_IN = ROOT / "data" / "raw" / "TCGA" / "tcga_clinical_full.csv"
OUT_DIR = ROOT / "data" / "raw" / "TCGA"

sys.path.insert(0, str(ROOT / "src"))
from case_id import resolve_case_id

GDC_CASES = "https://api.gdc.cancer.gov/cases"
BATCH_SIZE = 200
SEED = 42

def gdc_batch_lookup(submitter_ids: list[str], timeout: int = 30) -> dict:
    """批量查 submitter_id → {project_id, case_id, ...}"""
    if not submitter_ids:
        return {}
    filters = {
        "op": "in",
        "content": {"field": "cases.submitter_id", "value": submitter_ids}
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "case_id,submitter_id,project.project_id,project.name,diagnoses.primary_diagnosis",
        "format": "JSON",
        "size": str(len(submitter_ids)),
    }
    url = f"{GDC_CASES}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    results = {}
    for h in data.get("data", {}).get("hits", []):
        sid = h.get("submitter_id", "")
        proj = h.get("project", {}).get("project_id", "")
        results[sid] = {
            "project_id": proj,                         # e.g. TCGA-LUAD
            "institution_code": sid.split("-")[1],      # e.g. 05 (TSS 代码)
            "case_id": h.get("case_id", ""),
        }
    return results


def fetch_all_institutions(submitter_ids: list[str]) -> dict:
    """分批查所有 submitter_id, 带重试."""
    all_results = {}
    n_batches = (len(submitter_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(n_batches):
        batch = submitter_ids[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        for attempt in range(3):
            try:
                print(f"  GDC batch {bi+1}/{n_batches} ({len(batch)} ids) ...", end=" ", flush=True)
                res = gdc_batch_lookup(batch)
                all_results.update(res)
                print(f"✓ {len(res)} hits")
                break
            except Exception as e:
                print(f"✗ attempt {attempt+1}: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            print(f"  ⚠ batch {bi+1} 跳过 (3 次失败)")
        time.sleep(0.3)  # 礼貌性限速
    return all_results


def build_split(df: pd.DataFrame, group_col: str, test_frac: float = 0.3, seed: int = SEED) -> pd.DataFrame:
    """按机构分组, 切 train/test.

    策略: 按机构样本量排序, 从小到大累加直到累计样本量 ≥ 总样本量×test_frac 的那个机构
    归入 test. 这样 test set 机构少但样本量稳定。
    """
    rng = np.random.RandomState(seed)
    institutions = df[group_col].unique()
    rng.shuffle(institutions)

    n_total = len(df)
    n_test_target = int(n_total * test_frac)

    # greedy: 从小机构开始塞 test, 直到够了
    counts = df[group_col].value_counts()
    small_first = counts.sort_values().index.tolist()

    test_orgs = set()
    cum = 0
    for org in small_first:
        test_orgs.add(org)
        cum += counts[org]
        if cum >= n_test_target:
            break
    # 如果机构太少 (机构数 < 3), 退化成 sample-level split
    if len(test_orgs) == 0 or len(institutions) - len(test_orgs) < 2:
        print(f"  ⚠ 机构太少 ({len(institutions)}), 退化为 sample-level 随机 split")
        return pd.Series(["train"] * (len(df) - n_test_target) + ["test"] * n_test_target,
                         index=rng.permutation(df.index)).sort_index()

    return df[group_col].apply(lambda x: "test" if x in test_orgs else "train")


def main():
    print("=== 0. Load clinical ===")
    clin = pd.read_csv(CLINICAL_IN)
    print(f"  clinical: {clin.shape}")

    # submitter_id = case_id 去掉 patient barcode 后缀 (只留 TCGA-XX-XXXX)
    clin["submitter_id"] = clin["case_id"].str[:12]
    all_submitters = clin["submitter_id"].unique().tolist()
    print(f"  unique submitter_ids: {len(all_submitters)}")

    print("\n=== 1. Fetch institutions from GDC ===")
    inst_map = fetch_all_institutions(all_submitters)
    print(f"  GDC returned: {len(inst_map)} / {len(all_submitters)}")

    # unmatched
    unmatched = [s for s in all_submitters if s not in inst_map]
    if unmatched:
        print(f"  ⚠ 未匹配 ({len(unmatched)}): {unmatched[:5]}...")

    print("\n=== 2. Merge ===")
    inst_df = pd.DataFrame.from_dict(inst_map, orient="index")
    inst_df.index.name = "submitter_id"
    inst_df = inst_df.reset_index()

    merged = clin.merge(inst_df, on="submitter_id", how="left")

    # institution_code = TSS 代码 (TCGA-XX 中的 XX)
    merged["institution_code"] = merged["submitter_id"].str.split("-").str[1]

    print("\n=== 3. Institution summary ===")
    # 按 TSS 代码 + 项目分组
    summary = (merged.groupby(["project_id", "institution_code"])
               .size()
               .reset_index(name="n_samples")
               .sort_values("n_samples", ascending=False))
    print(f"  机构-项目组合数: {len(summary)}")
    print(f"  Top 10:\n{summary.head(10).to_string(index=False)}")

    # 只保留有 institution 的 (GDC 匹配到的)
    merged_gd = merged[merged["project_id"].notna()].copy()
    print(f"\n  有机构信息的样本: {len(merged_gd)} / {len(merged)}")

    print("\n=== 4. Build cross-institution split ===")
    # 按 institution_code 切 (更细, 约 200+ 机构), 或按 project_id 切 (粗, LUAD/LUSC 各一个)
    # 按 TSS institution_code 切最有意义 — 不同医院不同机构
    merged_gd["split_institution"] = build_split(merged_gd, "institution_code", test_frac=0.30)

    # 统计
    for split_name, sub in merged_gd.groupby("split_institution"):
        orgs = sub["institution_code"].nunique()
        print(f"  {split_name}: {len(sub)} samples, {orgs} institutions")

    # 保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "tcga_clinical_with_institution.csv"
    merged_gd.to_csv(out_csv, index=False)
    print(f"\n✓ 保存: {out_csv}")

    summary_csv = OUT_DIR / "institution_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"✓ 保存: {summary_csv}")

    # 快速预览下游用
    train = merged_gd[merged_gd["split_institution"] == "train"]["submitter_id"].tolist()
    test = merged_gd[merged_gd["split_institution"] == "test"]["submitter_id"].tolist()
    print(f"\n=== Split Summary ===")
    print(f"  Train: {len(train)} patients, {merged_gd[merged_gd['split_institution']=='train']['institution_code'].nunique()} institutions")
    print(f"  Test:  {len(test)} patients, {merged_gd[merged_gd['split_institution']=='test']['institution_code'].nunique()} institutions")
    print(f"\n✅ GDC institution split 完成! 下一步: 服务器上用这个 CSV 重跑 downstream DG.")


if __name__ == "__main__":
    main()
