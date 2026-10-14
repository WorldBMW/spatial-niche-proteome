"""服务器上直接跑: import load_omics + 简单 pearson (不依赖不存在的函数)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.chdir(os.path.dirname(__file__))

# 只 import 确定存在的函数
from downstream_omics import load_cptac_omics
import pandas as pd
import numpy as np

print("=== Step 1: load omics ===")
omics = load_cptac_omics(None)
print("loaded:", list(omics.keys()))
for k, v in omics.items():
    if hasattr(v, "shape"): print(f"  {k}: {v.shape}")

print("\n=== Step 2: load features CSV ===")
FEAT = "data/features/dinov2_vitl14_features_cptac/all_features_omics_aligned.csv"
feat = pd.read_csv(FEAT)
print(f"features: {feat.shape}")
print(f"feat cols head: {feat.columns[:5].tolist()}")

print("\n=== Step 3: case_id 交集 ===")
prot = omics["proteomics"]
# 找 patient/case id 列
prot_id_col = None
for c in prot.columns:
    if "patient" in c.lower() or c == "case_id" or c == "case":
        prot_id_col = c; break
# 也可能 index 就是 patient_id
if prot_id_col is None and prot.index.name and "patient" in prot.index.name.lower():
    prot = prot.reset_index()
    prot_id_col = prot.columns[0]

print(f"  proteomics id col: {prot_id_col}")
print(f"  proteomics shape: {prot.shape}")
print(f"  proteomics cols head: {prot.columns[:5].tolist()}")

if prot_id_col:
    prot[prot_id_col] = prot[prot_id_col].astype(str)
    feat_cases = set(feat["case_id"].astype(str))
    prot_cases = set(prot[prot_id_col])
    common = feat_cases & prot_cases
    print(f"  feat cases: {len(feat_cases)}, prot cases: {len(prot_cases)}, common: {len(common)}")

print("\n=== Step 4: simple pearson on 5 targets ===")
if prot_id_col and len(common) > 100:
    feat_sel = feat[feat["case_id"].isin(common)].sort_values("case_id").reset_index(drop=True)
    prot_sel = prot[prot[prot_id_col].isin(common)].sort_values(prot_id_col).reset_index(drop=True)
    
    # 确认行对齐
    assert len(feat_sel) == len(prot_sel), f"row mismatch {len(feat_sel)} vs {len(prot_sel)}"
    
    feat_cols = [c for c in feat_sel.columns if c != "case_id"]
    target_cols = [c for c in prot_sel.columns if c != prot_id_col][:5]
    
    print(f"  aligned: {len(feat_sel)} cases × {len(feat_cols)} features")
    print(f"  targets: {target_cols}")
    
    X = feat_sel[feat_cols].values.astype(np.float32)
    r_list = []
    for t in target_cols:
        y = prot_sel[t].values.astype(np.float32)
        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1) & (np.std(y) > 1e-8)
        n_valid = mask.sum()
        if n_valid > 10:
            r_vals = np.array([np.corrcoef(X[mask,i], y[mask])[0,1] for i in range(X.shape[1]) if np.std(X[mask,i]) > 1e-8], dtype=np.float32)
            # 处理 nan
            r_vals = r_vals[~np.isnan(r_vals)]
            if len(r_vals) > 0:
                r_max = np.max(np.abs(r_vals))
                r_mean = np.mean(r_vals)
                print(f"    {t[:55]}: n={n_valid}, max|r|={r_max:.3f}, mean r={r_mean:+.3f}")
                r_list.append((t, r_max, n_valid))
            else:
                print(f"    {t[:55]}: all r NaN (var too small?)")
        else:
            print(f"    {t[:55]}: too few valid ({n_valid})")
    
    if r_list:
        from scipy.stats import pearsonr as _pr
        print(f"\n=== Top hits (|r|>0.15) ===")
        strong = [(t, r, n) for t, r, n in r_list if r > 0.15]
        print(f"  targets with max|r|>0.15: {len(strong)}/{len(r_list)}")

print("\n=== DONE ===")
