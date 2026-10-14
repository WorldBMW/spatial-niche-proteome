import sys; sys.path.insert(0, 'src')
from downstream_omics import load_cptac_omics, unify_patient_ids, load_morphology_features
import numpy as np, pandas as pd

omics = load_cptac_omics(None, modalities=['clinical'])
omics = unify_patient_ids(omics)
clin = omics['clinical']
print('clinical shape:', clin.shape)

feat_df = load_morphology_features('results_vitl14_jbi/all_features_omics_aligned.csv')
case_ids = feat_df['case_id'].astype(str).tolist()
print(f'features n={len(case_ids)}')

c = clin.copy()
if 'case_id' not in c.columns and 'patient_id' in c.columns:
    c['case_id'] = c['patient_id'].astype(str)

t_col = 'Overall survival, days'
e_col = 'Survival status (1, dead; 0, alive)'
print(f't_col present: {t_col in c.columns}')
print(f'e_col present: {e_col in c.columns}')

c = c[c['case_id'].isin(case_ids)].copy()
print(f'after case_id filter: {len(c)}')
c['time'] = pd.to_numeric(c[t_col], errors='coerce')
c['event'] = pd.to_numeric(c[e_col], errors='coerce')
c = c[c['time'].notna() & c['event'].notna() & (c['time'] > 0)]
print(f'after time/event filter: {len(c)}')
ev = c['event'].value_counts().to_dict()
print(f'event distribution: {ev}')
print(f'time: mean={c["time"].mean():.1f}, median={c["time"].median():.1f}, min={c["time"].min()}, max={c["time"].max()}')
print(f'time=0 count: {(c["time"]==0).sum()}')
