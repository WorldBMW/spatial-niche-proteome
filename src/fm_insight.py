"""
fm_insight.py
=============
Foundation Model Topology Insight — JBI 一区第二个贡献 (Model Interpretation)。

对比 FM (DINOv2) vs ImageNet supervised (ResNet50) 在 WSI embedding 空间的本质差异。

核心分析:
  1. Embedding 空间拓扑: UMAP → silhouette score (LUAD vs LUSC 可分性)
  2. Niche 分布可分性: centroid pairwise cosine distance + within-cluster purity
  3. FM vs ImageNet 对比: kNN 近似 Hausdorff + niche centroid collapse

用法:
  python src/fm_insight.py \
      --backbone dinov2_vitl14 \
      [--comparison-backbone resnet50] \
      [--embedding-dir data/processed/dinov2_vitl14_embeddings] \
      [--clinical data/raw/TCGA/tcga_clinical_full.csv] \
      [--out results/dinov2_vitl14/insight/]
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# case_id 解析 (同项目内复用)
sys.path.insert(0, str(Path(__file__).parent))
from case_id import resolve_case_id  # noqa: E402

# ---------------------------------------------------------------------------
# 日志 & 常量
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fm_insight")

RANDOM_STATE = 2026
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
NICHE_PURITY_SAMPLE = 1000
HAUSDORFF_K = 3
COLLAPSE_COSINE_THRESHOLD = 0.3


# ===================================================================
# 1. 数据加载层
# ===================================================================
def load_embeddings_patient_mean(
    embed_dir: Path,
) -> Tuple[np.ndarray, List[str], int]:
    """
    加载一个 backbone 下所有 case 的 embedding, 取 mean 得到 patient-level representation.

    Returns
    -------
    patient_embed : np.ndarray, shape (N_patients, embed_dim)
    case_ids      : List[str]   与 patient_embed 行对齐的 case_id (已 resolve)
    embed_dim     : int
    """
    npy_files = sorted(
        f for f in embed_dir.glob("*.npy")
        if "_coords" not in f.name and "_probs" not in f.name
        and "_argmax" not in f.name
    )
    if not npy_files:
        raise FileNotFoundError(f"No embedding .npy found in {embed_dir}")

    embeds: List[np.ndarray] = []
    case_ids: List[str] = []
    for f in npy_files:
        raw = np.load(f)
        # 过滤掉极端 case (空 embedding)
        if raw.ndim != 2 or raw.shape[0] == 0:
            log.warning("skip %s: shape=%s", f.name, raw.shape)
            continue
        embeds.append(raw.mean(axis=0).astype(np.float32))
        case_ids.append(resolve_case_id(f.name))

    patient_embed = np.stack(embeds, axis=0)
    log.info("Loaded %d patients, embed_dim=%d from %s",
             len(case_ids), patient_embed.shape[1], embed_dir)
    return patient_embed, case_ids, patient_embed.shape[1]


def load_embeddings_patch_level(
    embed_dir: Path,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    加载 patch-level embedding (用于 niche purity 分析).

    Returns
    -------
    patch_embed_map : Dict[case_id, np.ndarray (N_patches, embed_dim)]
    embed_dim       : int
    """
    patch_map: Dict[str, np.ndarray] = {}
    npy_files = sorted(
        f for f in embed_dir.glob("*.npy")
        if "_coords" not in f.name and "_probs" not in f.name
        and "_argmax" not in f.name
    )
    dim = None
    for f in npy_files:
        raw = np.load(f)
        if raw.ndim != 2 or raw.shape[0] == 0:
            continue
        if dim is None:
            dim = raw.shape[1]
        patch_map[resolve_case_id(f.name)] = raw.astype(np.float32)
    log.info("Patch-level: %d cases, embed_dim=%d", len(patch_map), dim)
    return patch_map, dim  # type: ignore[return-value]


def load_kmeans(niche_dir: Path, backbone: str) -> "sklearn.cluster.KMeans":
    """加载已训练好的全局 KMeans 模型."""
    pkl_path = niche_dir / f"kmeans_{backbone}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"KMeans pickle not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        km = pickle.load(f)
    log.info("Loaded KMeans: %s (n_clusters=%d)", pkl_path, km.n_clusters)
    return km


def load_clinical(clinical_csv: Path) -> pd.DataFrame:
    """加载 clinical CSV, 统一 case_id 格式, 归一化 cohort (LUAD / LUSC)."""
    df = pd.read_csv(clinical_csv)
    if "case_id" not in df.columns:
        raise KeyError(f"clinical CSV must have 'case_id' column: {clinical_csv}")
    df = df.copy()
    df["case_id"] = df["case_id"].astype(str).str.upper()

    # 归一化 cohort 标签 → LUAD / LUSC
    if "cohort" in df.columns:
        df["cohort"] = df["cohort"].astype(str).str.upper()
        # 去掉 "TCGA-" 前缀 (如果有)
        df["cohort"] = df["cohort"].str.replace(r"^TCGA[-_]", "", regex=True)
        df["cohort"] = df["cohort"].str.replace(r"^CPTAC[-_]", "", regex=True)

    log.info("Loaded clinical: %d rows, cohorts=%s",
             len(df), df["cohort"].unique().tolist() if "cohort" in df.columns else "N/A")
    return df


# ===================================================================
# 2. Embedding 空间拓扑 — UMAP + silhouette
# ===================================================================
def run_umap_patient(
    patient_embed: np.ndarray,
    case_ids: List[str],
    cohorts: List[str],
    backbone: str,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, float]:
    """
    Patient-level UMAP 降维 + silhouette score (LUAD vs LUSC 可分性).

    Returns
    -------
    umap_df         : DataFrame (case_id, x, y, backbone, cohort)
    silhouette      : float (sklearn.metrics.silhouette_score on UMAP 2D)
    """
    import umap  # 延迟导入, 避免 import 失败时整个模块挂掉

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    # L2 normalize 再 cosine (UMAP 的 cosine metric 对零向量会报 warning)
    norms = np.linalg.norm(patient_embed, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    embed_normed = patient_embed / norms

    coords_2d = reducer.fit_transform(embed_normed)

    umap_df = pd.DataFrame({
        "case_id": case_ids,
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
        "backbone": backbone,
        "cohort": cohorts,
    })

    # silhouette 需要至少 2 个 cohort 且每个至少 2 个点
    cohort_arr = np.array(cohorts)
    unique_cohorts = np.unique(cohort_arr)
    if len(unique_cohorts) >= 2:
        try:
            sil = float(silhouette_score(
                coords_2d, cohort_arr, metric="euclidean"
            ))
        except Exception as e:
            log.warning("silhouette_score failed: %s", e)
            sil = float("nan")
    else:
        log.warning("Only %d cohort(s), silhouette=NA", len(unique_cohorts))
        sil = float("nan")

    log.info("[UMAP] %s: silhouette(LUAD/LUSC)=%.4f, 2D spread=[%.2f, %.2f] x [%.2f, %.2f]",
             backbone, sil, coords_2d[:, 0].min(), coords_2d[:, 0].max(),
             coords_2d[:, 1].min(), coords_2d[:, 1].max())
    return umap_df, sil


# ===================================================================
# 3. Niche 分布可分性
# ===================================================================
def avg_inter_niche_distance(km) -> float:
    """
    KMeans centroids 之间的平均 pairwise cosine distance.

    cosine distance = 1 - cosine similarity, 值域 [0, 2].
    值越大 → niche 之间分得越开.
    """
    centers = km.cluster_centers_  # (k, d)
    # L2 normalize
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    centers_n = centers / norms
    # cosine distance matrix
    sim = centers_n @ centers_n.T
    dist = 1.0 - sim  # [0, 2]
    # 只取下三角 (不含对角线)
    k = centers.shape[0]
    iu = np.triu_indices(k, k=1)
    avg_d = float(dist[iu].mean())
    log.info("avg_inter_niche_distance = %.4f (k=%d)", avg_d, k)
    return avg_d


def compute_niche_purity(
    patch_embed_map: Dict[str, np.ndarray],
    km,
    n_sample: int = NICHE_PURITY_SAMPLE,
    random_state: int = RANDOM_STATE,
) -> float:
    """
    随机抽 n_sample patches, 对每个 patch: 取其 KMeans cluster,
    计算该 cluster 内所有 patch 的 pairwise cosine similarity 均值.
    最终返回所有采样 patch 的这个 cluster 内相似度的平均值.

    值越高 → 同一 niche 内的 embedding 越"纯".
    """
    rng = np.random.RandomState(random_state)

    # 1. 聚合所有 patch embedding 和 argmax 标签 (不做 shuffle, 先串起来)
    all_embeds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for cid, emb in patch_embed_map.items():
        all_embeds.append(emb)
        all_labels.append(km.predict(emb))

    X = np.concatenate(all_embeds, axis=0)
    y = np.concatenate(all_labels, axis=0)
    n_total = X.shape[0]
    log.info("[niche_purity] total patches = %d, sampling %d", n_total, min(n_sample, n_total))

    if n_total == 0:
        return float("nan")

    idx = rng.choice(n_total, size=min(n_sample, n_total), replace=False)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    Xn = X / norms

    per_patch_similarity = []
    for i in idx:
        cluster_id = y[i]
        mask = (y == cluster_id)
        cluster_n = int(mask.sum())
        if cluster_n < 2:
            continue
        cluster_embeds = Xn[mask]
        # 该 patch 与 cluster 内其他 patch 的平均 cosine similarity
        sims = cluster_embeds @ Xn[i]  # (cluster_n,)
        # 去掉自己 (==1)
        mean_sim = float((sims.sum() - 1.0) / (cluster_n - 1))
        per_patch_similarity.append(mean_sim)

    if not per_patch_similarity:
        return float("nan")

    purity = float(np.mean(per_patch_similarity))
    log.info("avg_niche_purity = %.4f", purity)
    return purity


# ===================================================================
# 4. Hausdorff distance — DINOv2 manifold vs ResNet50 manifold
# ===================================================================
def _kNN_directed_hausdorff(
    A: np.ndarray,
    B: np.ndarray,
    k: int = HAUSDORFF_K,
    metric: str = "cosine",
) -> float:
    """
    kNN 近似的单向 Hausdorff distance:
      h(A → B) = max_{a in A} avg_{k-NN of a in B} distance

    用 NearestNeighbors 实现, 避免 O(|A|*|B|) 暴力.
    """
    if len(B) < 2:
        return float("nan")

    nn = NearestNeighbors(n_neighbors=min(k, len(B)), metric=metric)
    nn.fit(B)
    dists, _ = nn.kneighbors(A)  # (|A|, k)
    # cosine distance = 1 - cosine similarity, sklearn 的 cosine metric 返回的就是 cosine distance
    # 取每个 a 的 k-NN 平均距离, 再取 max
    per_a_mean = dists.mean(axis=1)
    return float(per_a_mean.max())


def hausdorff_between_backbones(
    embed_a: np.ndarray,
    embed_b: np.ndarray,
    k: int = HAUSDORFF_K,
    random_state: int = RANDOM_STATE,
) -> float:
    """
    两个 backbone 的 patient embedding manifold 之间的 Hausdorff distance.

    由于 DINOv2 (1024d) 和 ResNet50 (2048d) 维度不同, 无法直接在原始空间比较.
    本函数先分别对两个 backbone 的 patient embedding 做 PCA 降到 50 维,
    再在同一个 50 维空间计算 Hausdorff (cosine).

    返回值越大 → 两个 manifold 在低维主成分空间里相距越远.
    """
    from sklearn.decomposition import PCA

    pca_dim = 50
    pca_a = PCA(n_components=min(pca_dim, embed_a.shape[1], embed_a.shape[0] - 1),
                random_state=random_state)
    pca_b = PCA(n_components=min(pca_dim, embed_b.shape[1], embed_b.shape[0] - 1),
                random_state=random_state)

    Za = pca_a.fit_transform(embed_a).astype(np.float32)
    Zb = pca_b.fit_transform(embed_b).astype(np.float32)

    # 再做 L2 normalize, 用 cosine distance
    def _l2norm(Z: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        n = np.clip(n, 1e-8, None)
        return Z / n

    Za = _l2norm(Za)
    Zb = _l2norm(Zb)

    h_ab = _kNN_directed_hausdorff(Za, Zb, k=k, metric="cosine")
    h_ba = _kNN_directed_hausdorff(Zb, Za, k=k, metric="cosine")
    hausdorff = max(h_ab, h_ba)
    log.info("kNN-Hausdorff(dinov2↔resnet50) = %.4f (k=%d, PCA dim=%d)",
             hausdorff, k, pca_dim)
    return hausdorff


# ===================================================================
# 5. FM-specific insight — DINOv2 niche centroids 在 ResNet50 空间的塌缩
# ===================================================================
def analyze_niche_collapse(
    dinov2_patch_map: Dict[str, np.ndarray],
    resnet50_patch_map: Dict[str, np.ndarray],
    km_dinov2,
    km_resnet50,
    collapse_threshold: float = COLLAPSE_COSINE_THRESHOLD,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    对每个 DINOv2 niche (0..9):
      - 收集该 niche 下所有 patch 的 ResNet50 embedding
      - 在 ResNet50 空间计算它们的 pairwise cosine distance 均值
      - 若均值 < collapse_threshold → 标记为 "collapsed_in_resnet50"

    Returns DataFrame:
      niche_id, n_patches_in_niche, avg_resnet50_cosine_distance, collapsed
    """
    common_cases = sorted(set(dinov2_patch_map.keys()) & set(resnet50_patch_map.keys()))
    log.info("[niche_collapse] common cases with both backbones = %d", len(common_cases))
    if not common_cases:
        log.warning("No common cases between dinov2 & resnet50, skip collapse analysis.")
        return pd.DataFrame(columns=[
            "niche_id", "n_patches_in_niche",
            "avg_resnet50_cosine_distance", "collapsed",
        ])

    # DINOv2 space 里给每个 patch 分配 niche
    niche_to_resnet50_embeds: Dict[int, List[np.ndarray]] = {
        i: [] for i in range(km_dinov2.n_clusters)
    }

    for cid in common_cases:
        d_emb = dinov2_patch_map[cid]    # (N_p, 1024)
        r_emb = resnet50_patch_map[cid]  # (N_p, 2048)
        # 对齐: 同一个 case 的两个 backbone patch 顺序应该一致; 取 min
        n = min(d_emb.shape[0], r_emb.shape[0])
        if n == 0:
            continue
        labels = km_dinov2.predict(d_emb[:n])
        for i, lab in enumerate(labels):
            niche_to_resnet50_embeds[int(lab)].append(r_emb[i])

    rows = []
    for niche_id, emb_list in niche_to_resnet50_embeds.items():
        n_patches = len(emb_list)
        if n_patches < 2:
            rows.append({
                "niche_id": niche_id,
                "n_patches_in_niche": n_patches,
                "avg_resnet50_cosine_distance": float("nan"),
                "collapsed": False,
            })
            continue

        R = np.stack(emb_list, axis=0).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(R, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        Rn = R / norms
        # 平均 pairwise cosine distance = 1 - mean(cosine similarity)
        sim_mat = Rn @ Rn.T
        iu = np.triu_indices(n_patches, k=1)
        avg_sim = float(sim_mat[iu].mean())
        avg_dist = 1.0 - avg_sim

        rows.append({
            "niche_id": niche_id,
            "n_patches_in_niche": n_patches,
            "avg_resnet50_cosine_distance": avg_dist,
            "collapsed": avg_dist < collapse_threshold,
        })
        log.info("  niche[%d]  n_patches=%d  avg_resnet50_cos_dist=%.4f  collapsed=%s",
                 niche_id, n_patches, avg_dist, avg_dist < collapse_threshold)

    return pd.DataFrame(rows)


# ===================================================================
# 6. 顶层汇总
# ===================================================================
def run_backbone_analysis(
    backbone: str,
    embed_dir: Path,
    niche_dir: Path,
    clinical_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, object]:
    """
    单个 backbone 的完整分析. 输出 CSV 到 out_dir, 返回指标字典.
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("Analyzing backbone: %s", backbone)
    log.info("=" * 60)

    # --- 6.1 加载 patient-level embedding + KMeans ---
    patient_embed, case_ids, embed_dim = load_embeddings_patient_mean(embed_dir)
    km = load_kmeans(niche_dir, backbone)

    # --- 6.2 与 clinical 做 inner join ---
    clin = clinical_df.copy()
    clin["case_id"] = clin["case_id"].astype(str).str.upper()
    merged = clin[["case_id", "cohort"]].merge(
        pd.DataFrame({"case_id": [c.upper() for c in case_ids]}),
        on="case_id", how="inner",
    )
    keep_set = set(merged["case_id"].tolist())
    keep_idx = [i for i, c in enumerate(case_ids) if c.upper() in keep_set]

    if len(keep_idx) < 3:
        log.error("Too few matched patients (%d), abort %s", len(keep_idx), backbone)
        return {}

    patient_embed = patient_embed[keep_idx]
    case_ids_matched = [case_ids[i].upper() for i in keep_idx]
    # cohort 映射
    cohort_map = dict(zip(clin["case_id"], clin["cohort"]))
    cohorts = [cohort_map[c] for c in case_ids_matched]
    log.info("Matched patients = %d (LUAD=%d, LUSC=%d)",
             len(case_ids_matched),
             sum(1 for x in cohorts if x == "LUAD"),
             sum(1 for x in cohorts if x == "LUSC"))

    # --- 6.3 UMAP + silhouette ---
    umap_df, silhouette = run_umap_patient(
        patient_embed, case_ids_matched, cohorts, backbone
    )
    umap_path = out_dir / "fm_umap_coords.csv"
    umap_df.to_csv(umap_path, index=False)
    log.info("Saved UMAP coords → %s", umap_path)

    # --- 6.4 Niche 可分性 ---
    inter_dist = avg_inter_niche_distance(km)

    patch_map, _ = load_embeddings_patch_level(embed_dir)
    # 过滤到 clinical 有匹配的 case
    patch_map_filtered = {cid.upper(): emb
                          for cid, emb in patch_map.items()
                          if cid.upper() in keep_set}
    purity = compute_niche_purity(patch_map_filtered, km)

    metrics = {
        "backbone": backbone,
        "n_patients": len(case_ids_matched),
        "embed_dim": embed_dim,
        "silhouette_luad_lusc": round(silhouette, 6) if not np.isnan(silhouette) else None,
        "avg_inter_niche_distance": round(inter_dist, 6),
        "avg_niche_purity": round(purity, 6) if not np.isnan(purity) else None,
    }

    t1 = time.time()
    log.info("[%s] done in %.1fs  metrics=%s", backbone, t1 - t0, metrics)
    return metrics


# ===================================================================
# 7. 报告生成
# ===================================================================
def generate_report_md(
    comparison_df: pd.DataFrame,
    umap_paths: Dict[str, Path],
    collapse_df: Optional[pd.DataFrame],
    hausdorff_val: Optional[float],
    out_md: Path,
) -> None:
    """自动生成简要 Markdown 解读."""
    lines: List[str] = []
    lines.append("# Foundation Model Topology Insight Report\n")
    lines.append(f"_Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    lines.append("## 1. Backbone Comparison Table\n")
    lines.append(comparison_df.to_markdown(index=False))
    lines.append("")

    # --- Silhouette 解读 ---
    if "silhouette_luad_lusc" in comparison_df.columns:
        best_row = comparison_df.loc[comparison_df["silhouette_luad_lusc"].idxmax()]
        lines.append("## 2. Embedding Topology (UMAP silhouette)\n")
        lines.append(
            f"- **Best LUAD/LUSC separability**: `{best_row['backbone']}` "
            f"(silhouette={best_row['silhouette_luad_lusc']:.4f})\n"
        )
        lines.append(
            "> Silhouette ∈ [-1, 1]. 越接近 1 表示两个 cohort 在 2D UMAP "
            "空间分得越开; 接近 0 表示高度重叠.\n"
        )
    else:
        lines.append("## 2. Embedding Topology\n")
        lines.append("_No silhouette column available._\n")

    # --- Niche 可分性解读 ---
    if "avg_inter_niche_distance" in comparison_df.columns:
        lines.append("## 3. Niche Separability\n")
        for _, row in comparison_df.iterrows():
            lines.append(
                f"- **{row['backbone']}**: "
                f"`avg_inter_niche_distance`={row['avg_inter_niche_distance']:.4f}, "
                f"`avg_niche_purity`={row.get('avg_niche_purity', float('nan'))}\n"
            )
        lines.append(
            "> `avg_inter_niche_distance`: 10 个 KMeans centroid 之间的平均 cosine "
            "distance, 值越大 → niche 之间分得越开.\n"
            "> `avg_niche_purity`: 同一 niche 内 patch 平均 pairwise cosine "
            "similarity, 值越高 → niche 内部越纯.\n"
        )

    # --- Hausdorff ---
    if hausdorff_val is not None and not np.isnan(hausdorff_val):
        lines.append("## 4. Cross-Backbone Manifold Distance\n")
        lines.append(
            f"- **kNN-Hausdorff (PCA-50, cosine)** = `{hausdorff_val:.4f}`\n"
        )
        lines.append(
            "> 衡量 DINOv2 manifold 与 ResNet50 manifold 在保持主要方差的 "
            "低维空间中的几何差异. 值越大 → 两个 backbone 学到的 embedding 几何结构越不同.\n"
        )

    # --- Niche collapse ---
    if collapse_df is not None and not collapse_df.empty:
        lines.append("## 5. FM-Specific Niche Collapse in ResNet50\n")
        collapsed = collapse_df[collapse_df["collapsed"] == True]  # noqa: E712
        if collapsed.empty:
            lines.append(
                "**No DINOv2 niches collapsed in ResNet50 space.** "
                "所有 DINOv2 niche 在 ResNet50 空间中仍保持一定的分散度.\n"
            )
        else:
            lines.append(
                f"**{len(collapsed)}/{len(collapse_df)} DINOv2 niches "
                f"collapsed in ResNet50 space** (avg cosine distance < {COLLAPSE_COSINE_THRESHOLD}):\n"
            )
            lines.append(collapsed.to_markdown(index=False))
            lines.append("")
        lines.append(
            "> **Interpretation**: DINOv2 的 niche decomposition 能在自身 embedding 空间中 "
            "分离出 ResNet50 无法区分的细胞生态位. 塌缩的 niche 说明 ResNet50 把 "
            "DINOv2 学到的细粒度语义混在一起, 这可能是 FM 下游任务 (生存/突变预测) "
            "表现更好的拓扑学原因.\n"
        )

    # --- UMAP paths ---
    lines.append("## 6. Output Files\n")
    for bb, p in umap_paths.items():
        lines.append(f"- `{bb}` UMAP: `{p}`\n")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report saved → %s", out_md)


# ===================================================================
# 8. CLI
# ===================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Foundation Model Topology Insight (JBI contrib 2: Model Interpretation)"
    )
    ap.add_argument("--backbone", required=True,
                    help="Primary backbone name, e.g. dinov2_vitl14")
    ap.add_argument("--comparison-backbone", default=None,
                    help="Optional comparison backbone, e.g. resnet50. "
                         "If given, compute Hausdorff & niche collapse.")
    ap.add_argument("--embedding-dir", default=None,
                    help="Override embedding dir for --backbone. "
                         "Default: data/processed/{backbone}_embeddings")
    ap.add_argument("--comparison-embedding-dir", default=None,
                    help="Override embedding dir for --comparison-backbone.")
    ap.add_argument("--clinical", default=None,
                    help="Clinical CSV with case_id, cohort. "
                         "Default: data/raw/TCGA/tcga_clinical_full.csv")
    ap.add_argument("--out", default=None,
                    help="Output dir. Default: results/{backbone}/insight/")
    args = ap.parse_args()

    # --- resolve paths ---
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_proc = project_root / "data" / "processed"
    data_raw = project_root / "data" / "raw"

    embed_dir = Path(args.embedding_dir) if args.embedding_dir else (
        data_proc / f"{args.backbone}_embeddings"
    )
    niche_dir = data_proc / f"{args.backbone}_niches"
    out_dir = Path(args.out) if args.out else (
        project_root / "results" / args.backbone / "insight"
    )
    clinical_csv = Path(args.clinical) if args.clinical else (
        data_raw / "TCGA" / "tcga_clinical_full.csv"
    )

    # --- optional comparison backbone paths ---
    cmp_bb = args.comparison_backbone
    if cmp_bb is not None:
        cmp_embed_dir = Path(args.comparison_embedding_dir) if args.comparison_embedding_dir else (
            data_proc / f"{cmp_bb}_embeddings"
        )
        cmp_niche_dir = data_proc / f"{cmp_bb}_niches"
    else:
        cmp_embed_dir = None
        cmp_niche_dir = None

    # --- load clinical once ---
    if not clinical_csv.exists():
        log.error("clinical CSV not found: %s", clinical_csv)
        raise SystemExit(1)
    clinical_df = load_clinical(clinical_csv)

    # --- run primary backbone ---
    primary_metrics = run_backbone_analysis(
        backbone=args.backbone,
        embed_dir=embed_dir,
        niche_dir=niche_dir,
        clinical_df=clinical_df,
        out_dir=out_dir,
    )

    all_metrics = [primary_metrics]

    # --- comparison backbone ---
    hausdorff_val: Optional[float] = None
    collapse_df: Optional[pd.DataFrame] = None
    umap_paths: Dict[str, Path] = {
        args.backbone: out_dir / "fm_umap_coords.csv",
    }

    if cmp_bb is not None:
        if cmp_embed_dir is None or not cmp_embed_dir.exists():
            log.warning("Comparison embedding dir not found: %s, skip.", cmp_embed_dir)
        else:
            cmp_out_dir = out_dir.parent / cmp_bb / "insight"
            cmp_metrics = run_backbone_analysis(
                backbone=cmp_bb,
                embed_dir=cmp_embed_dir,
                niche_dir=cmp_niche_dir,
                clinical_df=clinical_df,
                out_dir=cmp_out_dir,
            )
            if cmp_metrics:
                all_metrics.append(cmp_metrics)
            umap_paths[cmp_bb] = cmp_out_dir / "fm_umap_coords.csv"

            # --- Hausdorff ---
            log.info("Computing Hausdorff distance between %s and %s ...",
                     args.backbone, cmp_bb)
            # 重新加载 patient-level embedding (因为 run_backbone_analysis 内部用完就丢)
            p_embed, p_ids, _ = load_embeddings_patient_mean(embed_dir)
            c_embed, c_ids, _ = load_embeddings_patient_mean(cmp_embed_dir)
            # 只保留 clinical 有 cohort 的 case
            clin = clinical_df.copy()
            clin["case_id"] = clin["case_id"].astype(str).str.upper()
            keep_cases = set(clin["case_id"].tolist())
            p_mask = [i for i, c in enumerate(p_ids) if c.upper() in keep_cases]
            c_mask = [i for i, c in enumerate(c_ids) if c.upper() in keep_cases]
            p_embed = p_embed[p_mask]
            c_embed = c_embed[c_mask]
            hausdorff_val = hausdorff_between_backbones(p_embed, c_embed)
            # 附加到 comparison row 里 (加一列)
            # 直接改 all_metrics 里 cmp 那条
            for m in all_metrics:
                if m.get("backbone") == cmp_bb:
                    m["hausdorff_to_resnet"] = round(hausdorff_val, 6)

            # --- Niche collapse ---
            log.info("Analyzing niche collapse (%s niches in %s space) ...",
                     args.backbone, cmp_bb)
            dinov2_patch_map, _ = load_embeddings_patch_level(embed_dir)
            resnet_patch_map, _ = load_embeddings_patch_level(cmp_embed_dir)
            km_primary = load_kmeans(niche_dir, args.backbone)
            km_cmp = load_kmeans(cmp_niche_dir, cmp_bb) if cmp_niche_dir.exists() else None
            # niche collapse 不需要 cmp 的 KMeans (只用 primary 的 KMeans 分 niche)
            collapse_df = analyze_niche_collapse(
                dinov2_patch_map=dinov2_patch_map,
                resnet50_patch_map=resnet_patch_map,
                km_dinov2=km_primary,
                km_resnet50=km_cmp,  # 传进去但不使用, 保持 API 对称
            )
            collapse_path = out_dir / f"fm_niche_collapse_vs_{cmp_bb}.csv"
            collapse_df.to_csv(collapse_path, index=False)
            log.info("Collapse table saved → %s", collapse_path)

    # --- 汇总 CSV ---
    comparison_df = pd.DataFrame(all_metrics)
    summary_path = out_dir / "fm_topology_comparison.csv"
    comparison_df.to_csv(summary_path, index=False)
    log.info("Summary table saved → %s", summary_path)

    # --- Markdown 报告 ---
    md_path = out_dir / "fm_insight_report.md"
    generate_report_md(
        comparison_df=comparison_df,
        umap_paths=umap_paths,
        collapse_df=collapse_df,
        hausdorff_val=hausdorff_val,
        out_md=md_path,
    )

    log.info("All done. Results in → %s", out_dir)


if __name__ == "__main__":
    main()
