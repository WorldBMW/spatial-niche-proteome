import sys; sys.path.insert(0, 'src')
from downstream_omics import load_cptac_omics, unify_patient_ids, load_morphology_features
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

omics = load_cptac_omics(None, modalities=['clinical'])
omics = unify_patient_ids(omics)
clin = omics['clinical']
feat_df = load_morphology_features('results_vitl14_jbi/all_features_omics_aligned.csv')
case_ids = feat_df['case_id'].astype(str).tolist()

c = clin.copy()
if 'case_id' not in c.columns and 'patient_id' in c.columns:
    c['case_id'] = c['patient_id'].astype(str)
c = c[c['case_id'].isin(case_ids)].copy()
c['time'] = pd.to_numeric(c['Overall survival, days'], errors='coerce')
c['event'] = pd.to_numeric(c['Survival status (1, dead; 0, alive)'], errors='coerce')
c = c[c['time'].notna() & c['event'].notna() & (c['time'] > 0)]

# 用前 10 个特征测试
X = np.random.randn(len(c), 10)
df = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
df['time'] = c['time'].values
df['event'] = c['event'].values

print(f'train data shape: {df.shape}')
try:
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    cph.fit(df, duration_col='time', event_col='event', show_progress=False)
    print('fit OK')
    pred = cph.predict_partial_hazard(df.drop(columns=['time','event']))
    print(f'pred type: {type(pred)}, shape: {pred.shape if hasattr(pred,"shape") else "?"}')
    print(f'pred head: {pred.iloc[:3].tolist() if hasattr(pred,"iloc") else pred[:3]}')
    # concordance_index
    cidx = concordance_index(df['time'].values, -pred.values.ravel(), df['event'].values)
    print(f'c-index: {cidx:.3f}')
except Exception as e:
    import traceback
    traceback.print_exc()
