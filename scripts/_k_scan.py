"""
k-scan: 在 CPTAC patch embeddings 上评估 KMeans k=5..15 的聚类质量
输出: results_vitl14_jbi/k_scan.csv (k, inertia, silhouette, calinski_harabasz)
"""
import numpy as np, glob, sys, os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

EMB_DIR = "data/processed/dinov2_vitl14_embeddings_cptac"
OUT = "results_vitl14_jbi/k_scan.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

files = sorted(glob.glob(os.path.join(EMB_DIR, "*.npy")))
print(f"[k-scan] {len(files)} WSI embedding files", flush=True)

parts = []
for f in files:
    e = np.load(f)
    parts.append(e.astype(np.float32))
emb = np.concatenate(parts, axis=0)
print(f"[k-scan] total patches: {emb.shape}  (~{emb.shape[0]*emb.shape[1]*4/1e9:.2f} GB float32)", flush=True)

# 标准化 (silhouette 对尺度敏感)
emb_s = StandardScaler().fit_transform(emb)

# silhouette 抽样子集, 控制计算量
SIL_SAMPLE = 120000
rng = np.random.RandomState(0)
if len(emb_s) > SIL_SAMPLE:
    idx = rng.choice(len(emb_s), SIL_SAMPLE, replace=False)
    emb_sil = emb_s[idx]
else:
    emb_sil = emb_s
print(f"[k-scan] silhouette 子集: {emb_sil.shape}", flush=True)

rows = []
for k in range(5, 16):
    km = KMeans(n_clusters=k, random_state=0, n_init=10, max_iter=300).fit(emb_s)
    sil = silhouette_score(emb_sil, km.predict(emb_sil), sample_size=25000, random_state=0)
    ch = calinski_harabasz_score(emb_sil, km.predict(emb_sil))
    rows.append((k, float(km.inertia_), float(sil), float(ch)))
    print(f"[k-scan] k={k}  inertia={km.inertia_:.3e}  silhouette={sil:.4f}  CH={ch:.1f}", flush=True)

np.savetxt(OUT, np.array(rows), delimiter=",",
           header="k,inertia,silhouette,calinski_harabasz", comments="")
print("[k-scan] saved ->", OUT, flush=True)