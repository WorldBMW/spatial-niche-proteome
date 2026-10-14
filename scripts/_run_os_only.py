"""单独跑 OS Cox (PCA/标准化已在 run_os_analysis 外层 fold 内完成, 防泄露)"""
import sys; sys.path.insert(0, 'src')
from downstream_omics import (
    load_cptac_omics, unify_patient_ids, load_morphology_features,
    build_morphology_matrix, build_covariates, run_os_analysis
)
from pathlib import Path
import pandas as pd

import os
_FEAT_CAND = [
    "data/features/dinov2_vitl14_features_cptac/all_features_omics_aligned.csv",  # 服务器
    "results_vitl14_jbi/all_features_omics_aligned.csv",                          # 本地
]
FEAT = next(p for p in _FEAT_CAND if os.path.exists(p))
OUT = "results_vitl14_jbi_r2/dn_ALL_rna500"  # fold 内预处理版 OS 结果

omics = load_cptac_omics(None, modalities=['clinical'])
omics = unify_patient_ids(omics)
feat_df = load_morphology_features(FEAT)

# ALL cohort (不过滤)
case_ids = feat_df['case_id'].astype(str).tolist()
X_all, feat_names = build_morphology_matrix(feat_df)
print(f'features: {X_all.shape}')

cov = build_covariates(omics['clinical'], case_ids)
# 注: 不再全集残差化; run_os_analysis 在每折训练集内做标准化+PCA
os_df = run_os_analysis(X_all, omics['clinical'], case_ids, cov=cov)
print(os_df.to_string(index=False))
if not os_df.empty:
    os_df.to_csv(Path(OUT) / 'ALL_os_cox.csv', index=False)
    print(f'saved → {OUT}/ALL_os_cox.csv')
