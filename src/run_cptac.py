"""
run_cptac.py
============
CPTAC-LUAD/LSCC 外部队列的形态学 pipeline:
  WSI -> patch embedding -> KMeans niche 分解 -> 230 维空间生态特征 -> 与多组学对齐

设计要点 (外部验证不污染 TCGA 池):
  - 独立输出目录 (tag 区分, 默认 cptac):
      data/processed/{backbone}_embeddings_{tag}/
      data/processed/{backbone}_niches_{tag}/   (含 kmeans_{backbone}.pkl)
      data/features/{backbone}_features_{tag}/all_features.csv
  - case_id 由文件名 {Case_ID}__{Slide_ID}.svs 解析为患者 ID (C3L-/C3N-), 与组学名册一致
  - niche KMeans 两种模式:
      默认: 在 CPTAC 形态学上自拟合 (无监督, 不接触任何蛋白标签 -> 无标签泄漏)
      --frozen-kmeans: 加载 TCGA 拟合好的质心, 冻结应用 (严格外部验证 / 敏感性分析)

典型用法:
  # GPU 服务器全量 (209 张, ViT-L)
  python src/run_cptac.py --backbone dinov2_vitl14
  # 严格冻结外部验证
  python src/run_cptac.py --backbone dinov2_vitl14 \
      --frozen-kmeans data/processed/dinov2_vitl14_niches/kmeans_dinov2_vitl14.pkl
  # CPU 冒烟 (少量片/少量 WSI; 验证接线用, 产物 tag=smoke 可删)
  python src/run_cptac.py --backbone resnet50 --tag smoke --limit 2 --max-patches 120
"""
import sys
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from case_id import resolve_case_id
import patch_embed
import niche_decomp
import features as feat_mod


OMICS_DIR = config.DATA_RAW / "CPTAC" / "omics"
ROSTER = OMICS_DIR / "patient_roster.csv"


def _dirs(backbone: str, tag: str):
    embed_dir = config.DATA_PROC / f"{backbone}_embeddings_{tag}"
    niche_dir = config.DATA_PROC / f"{backbone}_niches_{tag}"
    feat_dir = config.DATA_FEAT / f"{backbone}_features_{tag}"
    for d in (embed_dir, niche_dir, feat_dir):
        d.mkdir(parents=True, exist_ok=True)
    return embed_dir, niche_dir, feat_dir


# ---------------- Step 1: embed ----------------
def step_embed(backbone, tag, limit, max_patches, force):
    embed_dir, _, _ = _dirs(backbone, tag)
    wsi_files = sorted(config.CPTAC_WSI_DIR.glob("*.svs"))
    if limit:
        wsi_files = wsi_files[:limit]
    print(f"[embed] {len(wsi_files)} WSI  max_patches={max_patches}  -> {embed_dir}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform, embed_dim = patch_embed.load_backbone(backbone, device)
    if device == "cuda":
        batch_size = 192
    else:
        # CPU: ViT-L 1.22G 模型 batch=16, 小模型 batch=32
        batch_size = 16 if "large" in backbone or "vitl" in backbone or "vit_l" in backbone else 32
    print(f"[embed] device={device}  batch_size={batch_size}  backbone={backbone}  embed_dim={embed_dim}")

    ok = skip = fail = 0
    t0 = time.time()
    for i, svs in enumerate(wsi_files, 1):
        case_id = resolve_case_id(svs.stem)
        out_npy = embed_dir / f"{case_id}.npy"
        out_coords = embed_dir / f"{case_id}_coords.csv"
        if not force and out_npy.exists() and out_coords.exists():
            skip += 1
            continue
        try:
            patches, coords = patch_embed.extract_patches_from_svs(
                svs, config.PATCH_SIZE, config.MAGNIFICATION,
                config.TISSUE_THRESHOLD, max_patches)
            if len(coords) < 10:
                print(f"  [{i}/{len(wsi_files)}] SKIP {case_id}: {len(coords)} patches")
                fail += 1
                continue
            emb = patch_embed.embed_patches(patches, model, transform, device,
                                            batch_size=batch_size, use_amp=(device == "cuda"))
            np.save(out_npy, emb)
            pd.DataFrame(coords, columns=["x", "y"]).to_csv(out_coords, index=False)
            ok += 1
            print(f"  [{i}/{len(wsi_files)}] {case_id}: {len(coords)} patches "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(wsi_files)}] ERROR {case_id}: {e}", flush=True)
    print(f"[embed] done. ok={ok} skip={skip} fail={fail}  {(time.time()-t0)/60:.1f} min")


# ---------------- Step 2: niche ----------------
def step_niche(backbone, tag, frozen_kmeans, force, embed_tag=None):
    # 嵌入与 KMeans 无关: 冻结模式可复用自拟合那套嵌入 (--embed-tag), 不必重新 embed
    embed_dir, _, _ = _dirs(backbone, embed_tag or tag)
    _, niche_dir, _ = _dirs(backbone, tag)
    km_path = Path(frozen_kmeans) if frozen_kmeans else (niche_dir / f"kmeans_{backbone}.pkl")

    if frozen_kmeans:
        with open(km_path, "rb") as f:
            km = pickle.load(f)
        print(f"[niche] FROZEN KMeans loaded: {km_path} (n_clusters={km.n_clusters})")
        if km.n_clusters != config.N_NICHES:
            raise ValueError(f"冻结模型 n_clusters={km.n_clusters} != config.N_NICHES={config.N_NICHES}")
    elif km_path.exists() and not force:
        with open(km_path, "rb") as f:
            km = pickle.load(f)
        print(f"[niche] reuse existing KMeans: {km_path}")
    else:
        print(f"[niche] fitting KMeans on CPTAC morphology -> {km_path}")
        km = niche_decomp.fit_global_kmeans(embed_dir, config.N_NICHES, seed=config.SEED)
        with open(km_path, "wb") as f:
            pickle.dump(km, f)

    npy_files = sorted(f for f in embed_dir.glob("*.npy") if "_coords" not in f.name)
    ok = 0
    for f in npy_files:
        case_id = f.stem
        out_p = niche_dir / f"{case_id}_probs.npy"
        out_a = niche_dir / f"{case_id}_argmax.npy"
        if out_p.exists() and out_a.exists() and not force:
            ok += 1
            continue
        emb = np.load(f)
        probs, argmax = niche_decomp.decompose_one(case_id, emb, km)
        np.save(out_p, probs)
        np.save(out_a, argmax)
        ok += 1
    print(f"[niche] done. {ok}/{len(npy_files)} decomposed.")


# ---------------- Step 3: features ----------------
def step_features(backbone, tag, force):
    _, niche_dir, feat_dir = _dirs(backbone, tag)
    out_csv = feat_dir / "all_features.csv"
    if out_csv.exists() and not force:
        print(f"[features] exists: {out_csv}")
        return out_csv

    probs_files = sorted(niche_dir.glob("*_probs.npy"))
    rows = []
    for pf in probs_files:
        case_id = pf.name.replace("_probs.npy", "")
        probs = np.load(pf)
        argmax = np.load(niche_dir / f"{case_id}_argmax.npy")
        rows.append([case_id] + feat_mod.extract_one(probs, argmax).tolist())
    df = pd.DataFrame(rows, columns=["case_id"] + feat_mod.FEATURE_NAMES)
    df.to_csv(out_csv, index=False)
    print(f"[features] {len(df)} cases × {len(feat_mod.FEATURE_NAMES)} -> {out_csv}")
    return out_csv


# ---------------- Step 4: align to omics ----------------
def step_align(backbone, tag):
    _, _, feat_dir = _dirs(backbone, tag)
    feat = pd.read_csv(feat_dir / "all_features.csv", dtype={"case_id": str})
    roster = pd.read_csv(ROSTER, dtype={"patient_id": str})

    m = feat.merge(roster, left_on="case_id", right_on="patient_id", how="inner")
    print(f"[align] 形态特征 {len(feat)} 例 ∩ 组学名册 {len(roster)} 例 = {len(m)} 例")
    if len(m) == 0:
        print("  [WARN] 0 交集! 特征 case_id 示例:", feat.case_id.head().tolist())
        return
    print("  按癌种:", m["cohort"].value_counts().to_dict())
    for mod in ["proteomics", "phosphoproteomics", "transcriptomics", "somatic_mutation"]:
        col = f"has_{mod}"
        print(f"  有 {mod:18s}: {int(m[col].sum())}")

    out = feat_dir / "all_features_omics_aligned.csv"
    m.to_csv(out, index=False)
    print(f"[align] saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.DEFAULT_BACKBONE,
                    choices=list(config.BACKBONES.keys()))
    ap.add_argument("--tag", default="cptac")
    ap.add_argument("--step", default="all",
                    choices=["embed", "niche", "features", "align", "all"])
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张 WSI")
    ap.add_argument("--max-patches", type=int, default=config.MAX_PATCHES_PER_WSI)
    ap.add_argument("--frozen-kmeans", type=str, default=None,
                    help="TCGA 拟合的 kmeans pkl; 不给则在 CPTAC 上自拟合")
    ap.add_argument("--embed-tag", type=str, default=None,
                    help="niche 步复用的嵌入目录 tag (默认与 --tag 相同)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    steps = ["embed", "niche", "features", "align"] if args.step == "all" else [args.step]
    for s in steps:
        print(f"\n{'='*70}\n[CPTAC step: {s}] backbone={args.backbone} tag={args.tag}\n{'='*70}")
        if s == "embed":
            step_embed(args.backbone, args.tag, args.limit, args.max_patches, args.force)
        elif s == "niche":
            step_niche(args.backbone, args.tag, args.frozen_kmeans, args.force,
                       embed_tag=args.embed_tag)
        elif s == "features":
            step_features(args.backbone, args.tag, args.force)
        elif s == "align":
            step_align(args.backbone, args.tag)


if __name__ == "__main__":
    main()
