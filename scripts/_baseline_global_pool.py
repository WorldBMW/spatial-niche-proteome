"""
baseline: DINOv2 全局池化 slide embedding (不做生态位分解)
对 proteomics top2000 跑同款 nested-CV ElasticNet 关联
输出: results_vitl14_jbi/dn_baseline_globalpool/ALL_proteomics_results.csv
2026-10-17: 加 --adjust-subtype (与 downstream_omics CLI 同源逻辑) + --out 参数
"""
import sys, os, glob, argparse, numpy as np
sys.path.insert(0, "src")
from downstream_omics import (load_cptac_omics, unify_patient_ids,
                              build_covariates, run_modality, load_morphology_features)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results_vitl14_jbi_r3/dn_baseline_globalpool/ALL_proteomics_results.csv")
ap.add_argument("--adjust-subtype", action="store_true",
                help="加入 LUAD/LUSC 亚型协变量 (2026-10-17 新增, 此前从未真正调整过)")
args = ap.parse_args()
OUT = args.out

EMB_DIR = "data/processed/dinov2_vitl14_embeddings_cptac"
FEAT_CSV = "data/features/dinov2_vitl14_features_cptac/all_features_omics_aligned.csv"

# 1) mean-pool per WSI
files = sorted(glob.glob(os.path.join(EMB_DIR, "*.npy")))
print(f"[baseline] {len(files)} embedding files", flush=True)
Xs = [np.load(f).astype(np.float32).mean(axis=0) for f in files]
X = np.stack(Xs)                       # (n, 1024)
print(f"[baseline] X pooled: {X.shape}", flush=True)

# 2) case_id 顺序: 优先用 features CSV 的 case_id (与主实验一致)
feat = load_morphology_features(FEAT_CSV)
case_ids = feat["case_id"].astype(str).tolist()
# embedding 文件名 → case_id (去掉 _coords / 后缀)
fbase = [os.path.basename(f).replace(".npy", "") for f in files]
# 尝试直接匹配; 若不匹配则尝试去掉前缀 (如 __ 开头的 alt 命名)
used = []
for c in case_ids:
    cands = [i for i, b in enumerate(fbase) if b == c or b.endswith(c) or c.endswith(b)]
    if len(cands) == 1:
        used.append(cands[0])
    elif len(cands) > 1:
        used.append(cands[0])          # 同名取第一个
    else:
        used.append(-1)
print(f"[baseline] 直接匹配 {len([u for u in used if u>=0])}/{len(case_ids)}", flush=True)
if any(u < 0 for u in used):
    # fallback: 按嵌入文件自身顺序对齐到 omics
    print("[baseline] 部分未匹配 → 按 embedding 文件顺序", flush=True)
    emb_ids = [b for b in fbase]
    X_emb = X
    omics = load_cptac_omics(None, modalities=["proteomics"])
    omics = unify_patient_ids(omics)
    om_df = omics["proteomics"].copy()
    om_df = om_df[om_df["case_id"].isin(emb_ids)].reset_index(drop=True)
    om_ids = om_df["case_id"].astype(str).tolist()
    idx = {c: i for i, c in enumerate(emb_ids)}
    row = [idx[c] for c in om_ids]
    Xm = X_emb[row]
    cov = build_covariates(omics.get("clinical"), om_ids)
    cov_m = cov.values if cov is not None else None
else:
    Xm = X[used]
    omics = load_cptac_omics(None, modalities=["proteomics"])
    omics = unify_patient_ids(omics)
    om_df = omics["proteomics"].copy()
    om_df = om_df[om_df["case_id"].isin(case_ids)].reset_index(drop=True)
    om_ids = om_df["case_id"].astype(str).tolist()
    idx = {c: i for i, c in enumerate(case_ids)}
    row = [idx[c] for c in om_ids]
    Xm = Xm[row]
    cov = build_covariates(omics.get("clinical"), om_ids)
    cov_m = cov.values if cov is not None else None

print(f"[baseline] aligned X: {Xm.shape}, omics: {om_df.shape}", flush=True)

# ---- subtype 协变量 (2026-10-17: 与 downstream_omics.py CLI L1027 同源逻辑) ----
if args.adjust_subtype and "cohort" in feat.columns:
    sub = (feat.set_index("case_id")["cohort"].astype(str).str.lower()
           .map({"luad": 0.0, "lusc": 1.0}).rename("subtype"))
    cov = cov.join(sub, how="left")
    print(f"[baseline] subtype (LUAD=0/LUSC=1) 加入协变量: 非空 "
          f"{int(cov['subtype'].notna().sum())}/{len(cov)}", flush=True)
    cov_m = cov.values
    # NaN 行在折内残差化时会被 mask 排除 (downstream 已处理)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
r_df = run_modality(X=Xm, omics_df=om_df, cov=cov_m, modality_name="proteomics",
                    top_k_variance=2000, out_csv=OUT, seed=2026, n_seeds=3,
                    do_permutation=False,
                    pca_components=50)  # fold 内 PCA 降至 50 维, 匹配生态位特征维度并提速
print(f"[baseline] done, {len(r_df)} targets -> {OUT}", flush=True)