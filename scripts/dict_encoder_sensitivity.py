# -*- coding: utf-8 -*-
"""
dict_encoder_sensitivity.py — 第二字典 (seed 敏感性) + 第二编码器 (ResNet50) 实验

目的:
  1. 字典抽样方差: 同一 backbone, 不同 k-means seed → 下游命中数变化
  2. 编码器鲁棒性: DINOv2 vs ResNet50 → 下游命中数变化

数据: TCGA WSI patches (data/processed/tcga_wsi_patches/*.npy) + TCGA 突变/临床
输出: results_dict_encoder_sensitivity/ 下的对比 CSV

用法:
  python scripts/dict_encoder_sensitivity.py --backbones dinov2_vitl14 resnet50 --seeds 2026 2027
  python scripts/dict_encoder_sensitivity.py --backbones dinov2_vitl14 --seeds 2026 2027 2028
"""
import sys, os, argparse, time, json, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from copy import deepcopy

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config

PATCH_DIR = config.DATA_PROC / "tcga_wsi_patches"
OUT_DIR = config.ROOT / "results_dict_encoder_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_tcga_case_ids():
    """从 tcga_wsi_files.csv 获取 GDC case UUID → TCGA patient ID 映射。
    npy 文件名 = GDC case UUID (case_id 列)。
    TCGA patient ID = file_name 的前 3 段 (e.g. TCGA-67-3773)."""
    fp = config.DATA_RAW / "TCGA" / "tcga_wsi_files.csv"
    if not fp.exists():
        print(f"[WARN] {fp} 不存在, 用 UUID 作 case_id")
        return None
    df = pd.read_csv(fp)
    mapping = {}
    for _, row in df.iterrows():
        gdc_uuid = str(row["case_id"])  # npy 文件名
        fname = str(row["file_name"])
        parts = fname.split(".")[0].split("-")
        if len(parts) >= 3:
            tcga_id = "-".join(parts[:3])  # TCGA-67-3773
            mapping[gdc_uuid] = tcga_id
    print(f"[mapping] {len(mapping)} GDC UUID → TCGA IDs")
    return mapping


def embed_all_patches(backbone: str, force: bool = False):
    """对所有 TCGA patches 做 backbone embedding, 缓存到本地。"""
    cache_fp = OUT_DIR / f"embeddings_{backbone}.npz"
    mapping = get_tcga_case_ids()

    if cache_fp.exists() and not force:
        print(f"[embed] 加载缓存: {cache_fp}")
        data = np.load(cache_fp, allow_pickle=True)
        case_ids = list(data["case_ids"])
        embeddings = list(data["embeddings"])
        # 应用 UUID → TCGA ID 映射 (缓存可能存的是 UUID)
        if mapping:
            case_ids = [mapping.get(cid, cid) for cid in case_ids]
        return case_ids, embeddings

    import patch_embed
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform, embed_dim = patch_embed.load_backbone(backbone, device=device)
    model.eval()

    mapping = get_tcga_case_ids()
    npy_files = sorted(PATCH_DIR.glob("*.npy"))
    print(f"[embed] {backbone}: {len(npy_files)} WSI, device={device}, embed_dim={embed_dim}")

    case_ids = []
    embeddings = []
    t0 = time.time()
    for i, fp in enumerate(npy_files):
        uuid = fp.stem
        case_id = mapping.get(uuid, uuid) if mapping else uuid
        patches_np = np.load(fp)  # (N, 224, 224, 3) uint8
        if len(patches_np) == 0:
            continue
        # numpy uint8 → PIL Image
        patches = [Image.fromarray(p) for p in patches_np]
        emb = patch_embed.embed_patches(patches, model, transform, device, batch_size=128)
        case_ids.append(case_id)
        embeddings.append(emb.astype(np.float32))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(npy_files)}] {case_id} {emb.shape}, elapsed={time.time()-t0:.0f}s")

    # 保存缓存
    np.savez_compressed(cache_fp,
                        case_ids=np.array(case_ids, dtype=object),
                        embeddings=np.array(embeddings, dtype=object))
    print(f"[embed] 完成: {len(case_ids)} WSI, 缓存到 {cache_fp}, 耗时 {time.time()-t0:.0f}s")
    return case_ids, embeddings


def fit_kmeans_and_decompose(case_ids, embeddings, n_clusters=10, seed=2026):
    """聚合所有 patch embedding → k-means → 每 WSI 软分配。"""
    from sklearn.cluster import KMeans

    all_feats = np.concatenate(embeddings, axis=0)
    print(f"[kmeans seed={seed}] 拟合 {all_feats.shape[0]} patches, {all_feats.shape[1]} dims")
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    km.fit(all_feats)

    # 每 WSI 软分配
    probs_list = []
    for emb in embeddings:
        dists = km.transform(emb)
        inv = 1.0 / (dists + 1e-6)
        probs = inv / inv.sum(axis=1, keepdims=True)
        probs_list.append(probs.astype(np.float32))
    return km, probs_list


def extract_features_from_probs(case_ids, probs_list, n_clusters=10):
    """从软分配提取 230 维特征: abund (10) + corr (45) + 多样性 (14) + 拓扑 (100) + Ripley (40) + 界面 (12) + CAF (10).
    简化版: 只用 abund + corr + 基本统计, 足够做敏感性对比。"""
    records = []
    for cid, probs in zip(case_ids, probs_list):
        abund = probs.mean(axis=0)  # (10,)
        # corr 矩阵上三角
        corr = np.corrcoef(probs.T)  # (10, 10)
        triu = corr[np.triu_indices(n_clusters, k=1)]  # (45,)
        # 多样性
        entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        diversity = [entropy.mean(), entropy.std(), entropy.max(), entropy.min()]
        # 每个 niche 的 patch 数比例
        argmax = probs.argmax(axis=1)
        niche_counts = np.bincount(argmax, minlength=n_clusters) / len(argmax)
        # 拼接
        feat = np.concatenate([abund, triu, diversity, niche_counts])
        records.append({"case_id": cid, **{f"feat_{j}": v for j, v in enumerate(feat)}})
    df = pd.DataFrame(records)
    return df


def load_tcga_targets():
    """加载 TCGA 突变 + TMB 作为下游二分类目标。"""
    clin = pd.read_csv(config.DATA_RAW / "TCGA" / "tcga_clinical_full.csv")
    clin["case_id"] = clin["case_id"].astype(str)
    targets = {}
    # TMB_high
    if "TMB_high" in clin.columns:
        targets["TMB_high"] = clin.set_index("case_id")["TMB_high"]
    # 突变基因
    mut_cols = [c for c in clin.columns if c in ["EGFR", "KRAS", "KEAP1", "TP53", "ALK", "BRAF", "MET", "PIK3CA"]]
    for col in mut_cols:
        targets[col] = clin.set_index("case_id")[col]
    return targets


def run_downstream_classification(feat_df, targets, n_splits=5, n_seeds=3):
    """对每个 target 做 ElasticNet 分类, 返回 q<0.05 的命中数。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    # 去重: 同一 patient 多个 WSI → 取平均
    feat_df = feat_df.groupby("case_id").mean().reset_index()

    X = feat_df.drop(columns=["case_id"]).values.astype(float)
    case_ids = feat_df["case_id"].values

    results = []
    for tname, tseries in targets.items():
        # 对齐
        y = tseries.reindex(case_ids)
        valid = ~y.isna()
        if valid.sum() < 30:
            continue
        y_vals = y[valid].values.astype(int)
        X_vals = X[valid]
        if len(np.unique(y_vals)) < 2:
            continue
        # 5-fold CV AUC
        aucs = []
        for seed in range(2026, 2026 + n_seeds):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for tr_idx, va_idx in skf.split(X_vals, y_vals):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_vals[tr_idx])
                X_va = scaler.transform(X_vals[va_idx])
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
                clf.fit(X_tr, y_vals[tr_idx])
                prob = clf.predict_proba(X_va)[:, 1]
                if len(np.unique(y_vals[va_idx])) < 2:
                    continue
                aucs.append(roc_auc_score(y_vals[va_idx], prob))
        if not aucs:
            continue
        mean_auc = np.mean(aucs)
        # 置换检验
        rng = np.random.RandomState(42)
        perm_aucs = []
        for _ in range(200):
            y_perm = rng.permutation(y_vals)
            pa = []
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2026)
            for tr_idx, va_idx in skf.split(X_vals, y_perm):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_vals[tr_idx])
                X_va = scaler.transform(X_vals[va_idx])
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=2026)
                clf.fit(X_tr, y_perm[tr_idx])
                prob = clf.predict_proba(X_va)[:, 1]
                if len(np.unique(y_perm[va_idx])) < 2:
                    continue
                pa.append(roc_auc_score(y_perm[va_idx], prob))
            if pa:
                perm_aucs.append(np.mean(pa))
        p_perm = np.mean([1 for a in perm_aucs if a >= mean_auc]) if perm_aucs else 1.0
        results.append({"target": tname, "n": int(valid.sum()),
                        "auc": round(mean_auc, 4), "p_perm": round(p_perm, 4),
                        "significant": p_perm < 0.05})
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["dinov2_vitl14", "resnet50"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    ap.add_argument("--force-embed", action="store_true")
    args = ap.parse_args()

    targets = load_tcga_targets()
    print(f"[targets] {len(targets)} 个: {list(targets.keys())}")

    all_results = []

    for backbone in args.backbones:
        print(f"\n{'='*60}\n[Backbone] {backbone}\n{'='*60}")
        case_ids, embeddings = embed_all_patches(backbone, force=args.force_embed)

        for seed in args.seeds:
            print(f"\n--- {backbone} seed={seed} ---")
            km, probs_list = fit_kmeans_and_decompose(case_ids, embeddings, seed=seed)
            feat_df = extract_features_from_probs(case_ids, probs_list)
            print(f"  features: {feat_df.shape}")

            res = run_downstream_classification(feat_df, targets)
            if res.empty:
                n_sig = 0
                print("  无有效 target")
            else:
                n_sig = int(res["significant"].sum())
                print(f"  命中数 (p<0.05): {n_sig}/{len(res)}")
                print(res.to_string(index=False))

            res["backbone"] = backbone
            res["seed"] = seed
            all_results.append(res)

            # 保存 features
            feat_df.to_csv(OUT_DIR / f"features_{backbone}_seed{seed}.csv", index=False)

    # 汇总
    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        summary.to_csv(OUT_DIR / "dict_encoder_sensitivity_summary.csv", index=False)

        # 透视表: backbone × seed → 命中数
        if "significant" in summary.columns:
            pivot = summary.groupby(["backbone", "seed"])["significant"].sum().reset_index()
            pivot.columns = ["backbone", "seed", "n_hits"]
        else:
            pivot = pd.DataFrame(columns=["backbone", "seed", "n_hits"])
    else:
        summary = pd.DataFrame()
        pivot = pd.DataFrame(columns=["backbone", "seed", "n_hits"])

    print("\n" + "=" * 60)
    print("字典/编码器敏感性汇总")
    print("=" * 60)
    print(pivot.to_string(index=False))
    pivot.to_csv(OUT_DIR / "dict_encoder_sensitivity_pivot.csv", index=False)
    print(f"\n结果保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
