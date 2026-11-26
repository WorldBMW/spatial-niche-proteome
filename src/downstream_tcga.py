"""P1: TCGA RNA-seq 下游预测 driver (跨队列验证 trans 主导性)。

复用 downstream_omics 的 run_modality / build_morphology_matrix / aggregate_feature_importance。

用法:
  python src/downstream_tcga.py \
    --features data/features/dinov2_vitl14_features_tcga_ucec/all_features.csv \
    --rnaseq data/raw/TCGA/rnaseq/tcga_ucec_rnaseq.csv \
    --subset only-trans --out-dir results_vitl14_jbi_pancancer/p1_tcga_rna/ucec_onlytrans
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from downstream_omics import (SEED, aggregate_feature_importance,
                              build_morphology_matrix, run_modality)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--rnaseq", required=True)
    ap.add_argument("--subset", default="full",
                    choices=["full", "no-trans", "only-trans", "only-abund",
                             "only-ripley", "only-corr", "matched-dim"])
    ap.add_argument("--shuffle-trans", action="store_true")
    ap.add_argument("--targets-top-k", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.features).parents[0].name.replace('dinov2_vitl14_features_', '')}_{args.subset}"
    print(f"\n{'='*70}\n[P1 TCGA RNA-seq] {tag}\n{'='*70}")

    # 1. features + rnaseq, inner join on case_id
    feat_df = pd.read_csv(args.features)
    rna = pd.read_csv(args.rnaseq)
    rna["case_id"] = rna["case_id"].astype(str)
    feat_df["case_id"] = feat_df["case_id"].astype(str)
    ids = set(feat_df["case_id"]) & set(rna["case_id"])
    feat_df = feat_df[feat_df["case_id"].isin(ids)].reset_index(drop=True)
    rna = rna[rna["case_id"].isin(ids)].reset_index(drop=True)
    if len(feat_df) < args.min_samples:
        print(f"[skip] 交集仅 {len(feat_df)} < {args.min_samples}, 跳过")
        return
    print(f"[align] 交集 {len(feat_df)} cases, {rna.shape[1]-1} genes")

    # 2. 特征矩阵 (含消融子集 / shuffle 零假设)
    X_all, feat_names = build_morphology_matrix(
        feat_df, feature_subset=args.subset, shuffle_trans=args.shuffle_trans)

    # 3. 跑 RNA-seq 预测 (无协变量, scipy p -> BH-FDR)
    r_df = run_modality(
        X=X_all, omics_df=rna, cov=None, modality_name="rnaseq",
        top_k_variance=args.targets_top_k,
        out_csv=str(out_dir / "rnaseq_results.csv"),
        n_seeds=args.seeds, feature_names=feat_names, do_permutation=False)

    # 4. 特征重要性
    if not r_df.empty:
        imp = aggregate_feature_importance(X_all, rna,
                                           target_cols=r_df["target"].tolist(),
                                           modality_name="rnaseq", seed=SEED)
        if not imp.empty:
            imp.to_csv(out_dir / "rnaseq_importance.csv", index=False)
            print(f"  特征重要性 top 5: {imp.head(5)['feature'].tolist()}")

    sig_n = int((r_df["q_value"] < 0.05).sum()) if "q_value" in r_df else 0
    print(f"\n[汇总] {tag}: 显著 {sig_n}/{len(r_df)} (q<0.05)")


if __name__ == "__main__":
    main()
