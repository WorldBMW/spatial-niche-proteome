"""验证 robust loader"""
import sys; sys.path.insert(0,'src')
from downstream_omics import load_cptac_omics, robust_read_csv
import pandas as pd, numpy as np

# 单独测 lusc_prot
df = robust_read_csv('data/raw/CPTAC/omics/lusc_proteomics.csv')
print(f'lusc_prot shape: {df.shape}')
print(f'patient_id head: {df["patient_id"].tolist()[:5]}')
print(f'patient_id dtype: {df["patient_id"].dtype}')

# 测交集
feat = pd.read_csv('results_vitl14_jbi/all_features_omics_aligned.csv')
feat_ids = set(feat['case_id'].astype(str))

luad = robust_read_csv('data/raw/CPTAC/omics/luad_proteomics.csv')
prot_ids_all = set(luad['patient_id'].astype(str)) | set(df['patient_id'].astype(str))
print(f'\nluad prot rows: {len(luad)}, lusc prot rows: {len(df)}')
print(f'prot all ids: {len(prot_ids_all)}')
common = feat_ids & prot_ids_all
print(f'feat ∩ prot_all: {len(common)}')

# 还需要去掉 .N/.T 后缀吗？
prot_ids_nosuf = {x.split('.')[0] for x in prot_ids_all}
common2 = feat_ids & prot_ids_nosuf
print(f'feat ∩ prot_nosuf: {len(common2)}')

if common2 > common:
    print('→ 需要去掉 .N/.T 后缀')
    feat_nosuf = {x.split('.')[0] for x in feat_ids}
    print(f'feat_nosuf ∩ prot_nosuf: {len(feat_nosuf & prot_ids_nosuf)}')
