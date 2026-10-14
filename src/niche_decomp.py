"""
niche_decomp.py
===============
用 patch_embed.py 输出的 (N, embed_dim) embedding → KMeans → 10 niche 软分配概率。

输出：
  {case_id}_probs.npy  shape=(N_patches, 10)  软分配概率
  {case_id}_argmax.npy shape=(N_patches,)       硬分配标签
  kmeans_{backbone}.pkl                         全局 KMeans 模型
"""
import sys, os, argparse, pickle, glob
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
import config
from case_id import is_uuid


def fit_global_kmeans(embed_dir: Path, n_clusters: int,
                      n_patches_max: int = 50000, seed: int = 2026):
    """聚合所有 WSI 的 embedding → 全局 KMeans。"""
    npy_files = sorted(embed_dir.glob("*.npy"))
    npy_files = [f for f in npy_files if "_coords" not in f.name]
    if not npy_files:
        raise FileNotFoundError(f"No embeddings in {embed_dir}")

    all_feats = []
    rng = np.random.RandomState(seed)
    for f in npy_files:
        feats = np.load(f)
        n = min(len(feats), n_patches_max // len(npy_files) + 1)
        idx = rng.choice(len(feats), size=n, replace=False)
        all_feats.append(feats[idx])
    X = np.concatenate(all_feats, axis=0)
    print(f"[kmeans] fitting on {X.shape[0]} patches, {X.shape[1]} dims")

    km = KMeans(n_clusters=n_clusters, n_init=config.KMEANS_N_INIT,
                random_state=seed, verbose=0)
    km.fit(X)
    return km


def decompose_one(case_id: str, embed: np.ndarray, km: KMeans):
    """单病例：embedding → soft/hard niche。"""
    dists = km.transform(embed)           # (N, 10)
    inv = 1.0 / (dists + 1e-6)
    probs = inv / inv.sum(axis=1, keepdims=True)   # softmax-like
    argmax = km.predict(embed)
    return probs.astype(np.float32), argmax.astype(np.int16)


def run(backbone: str = None):
    backbone = backbone or config.DEFAULT_BACKBONE
    embed_dir = config.embed_cache_dir(backbone)
    out_dir = config.niche_cache_dir(backbone)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 训练全局 KMeans（跨所有队列所有 WSI）
    km_path = out_dir / f"kmeans_{backbone}.pkl"
    if km_path.exists():
        with open(km_path, "rb") as f:
            km = pickle.load(f)
        print(f"[kmeans] loaded existing model: {km_path}")
    else:
        km = fit_global_kmeans(embed_dir, config.N_NICHES)
        with open(km_path, "wb") as f:
            pickle.dump(km, f)
        print(f"[kmeans] saved to {km_path}")

    # 2. 逐 WSI 分解（跳过旧的 UUID 格式 .npy — 无效数据）
    npy_files = sorted([f for f in embed_dir.glob("*.npy") if "_coords" not in f.name])
    ok = 0
    skipped_uuid = 0
    for f in tqdm(npy_files, desc=f"niche[{backbone}]"):
        case_id = f.stem
        # 跳过旧 UUID 格式（服务器上已有的无效数据）
        if is_uuid(case_id):
            skipped_uuid += 1
            continue
        out_p = out_dir / f"{case_id}_probs.npy"
        out_a = out_dir / f"{case_id}_argmax.npy"
        if out_p.exists() and out_a.exists():
            ok += 1
            continue
        embed = np.load(f)
        probs, argmax = decompose_one(case_id, embed, km)
        np.save(out_p, probs)
        np.save(out_a, argmax)
        ok += 1
    if skipped_uuid > 0:
        print(f"[niche] skipped {skipped_uuid} UUID-format .npy (旧数据)")
    print(f"[niche] done. {ok}/{len(npy_files)} decomposed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.DEFAULT_BACKBONE,
                    choices=list(config.BACKBONES.keys()))
    args = ap.parse_args()
    run(args.backbone)
