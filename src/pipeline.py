"""
pipeline.py
===========
一键跑通 experiment2 全流程：
  Step 1: embed      — patch embedding (DINOv2/ResNet50/...)
  Step 2: niche     — KMeans 10 niche 分解
  Step 3: features   — 230 维空间生态特征
  Step 4: kg         — Niche-Gene 条件互信息异构图 (JBI Knowledge Representation)
  Step 5: insight    — FM vs ImageNet embedding 拓扑对比 (JBI Model Interpretation)
  Step 6: downstream — 分子/生存预测 + 外部验证 + Holm 校正

用法：
  python src/pipeline.py --step all --backbone dinov2_vitl14
  python src/pipeline.py --step kg --backbone dinov2_vitl14
  python src/pipeline.py --step all --all-backbones
"""
import sys, argparse, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config

STEPS = ["embed", "niche", "features", "kg", "insight", "downstream"]


def run_step(step: str, backbone: str, force: bool = False):
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"[STEP {step}]  backbone={backbone}  force={force}")
    print(f"{'='*70}")

    # --- embed ---
    if step == "embed":
        import patch_embed
        out_dir = config.embed_cache_dir(backbone)
        existing = len(list(out_dir.glob("*.npy"))) if out_dir.exists() else 0
        if existing > 0 and not force:
            print(f"  [SKIP] {existing} embeddings already exist at {out_dir}")
        else:
            patch_embed.run(backbone=backbone)

    # --- niche ---
    elif step == "niche":
        import niche_decomp
        out_dir = config.niche_cache_dir(backbone)
        existing = len(list(out_dir.glob("*_probs.npy"))) if out_dir.exists() else 0
        if existing > 0 and not force:
            print(f"  [SKIP] {existing} niche decomps already exist")
        else:
            niche_decomp.run(backbone=backbone)

    # --- features ---
    elif step == "features":
        import features
        out_csv = config.features_dir(backbone) / "all_features.csv"
        if out_csv.exists() and not force:
            print(f"  [SKIP] {out_csv} already exists")
        else:
            features.run(backbone=backbone)

    # --- kg ---
    elif step == "kg":
        import niche_gene_kg
        out_dir = config.kg_dir(backbone)
        out_edges = out_dir / "niche_gene_edges.csv"
        if out_edges.exists() and not force:
            print(f"  [SKIP] {out_edges} already exists")
        else:
            feat_csv = config.features_dir(backbone) / "all_features.csv"
            clin_csv = config.DATA_RAW / "TCGA" / "tcga_clinical_full.csv"
            niche_gene_kg.main_func(
                features_csv=feat_csv,
                clinical_csv=clin_csv,
                out_dir=out_dir,
            ) if hasattr(niche_gene_kg, "main_func") else None
            # fallback: 直接用 CLI 入口
            if not hasattr(niche_gene_kg, "main_func"):
                import subprocess
                cmd = [sys.executable, str(Path(__file__).parent / "niche_gene_kg.py"),
                       "--features", str(feat_csv),
                       "--clinical", str(clin_csv),
                       "--out", str(out_dir)]
                subprocess.run(cmd, check=True)

    # --- insight ---
    elif step == "insight":
        import fm_insight
        out_dir = config.insight_dir(backbone)
        out_csv = out_dir / "fm_topology_comparison.csv"
        if out_csv.exists() and not force:
            print(f"  [SKIP] {out_csv} already exists")
        else:
            fm_insight.run_single(backbone=backbone, out_dir=out_dir) \
                if hasattr(fm_insight, "run_single") else None
            if not hasattr(fm_insight, "run_single"):
                import subprocess
                cmd = [sys.executable, str(Path(__file__).parent / "fm_insight.py"),
                       "--backbone", backbone,
                       "--out", str(out_dir)]
                subprocess.run(cmd, check=True)

    # --- downstream ---
    elif step == "downstream":
        import downstream
        out_csv = config.RESULTS / backbone / "downstream_summary.csv"
        if out_csv.exists() and not force:
            print(f"  [SKIP] {out_csv} already exists")
        else:
            downstream.run(backbone=backbone)

    # --- all ---
    elif step == "all":
        for s in STEPS:
            run_step(s, backbone, force=force)

    else:
        print(f"[WARN] unknown step: {step}. Valid: {STEPS}")
        return

    dt = time.time() - t0
    print(f"\n[STEP {step}] done in {dt/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["all"] + STEPS)
    ap.add_argument("--backbone", default=config.DEFAULT_BACKBONE,
                    choices=list(config.BACKBONES.keys()))
    ap.add_argument("--all-backbones", action="store_true",
                    help="依次跑所有 4 个 backbone")
    ap.add_argument("--force", action="store_true",
                    help="强制重跑（覆盖已有输出）")
    args = ap.parse_args()

    if args.all_backbones:
        for bb in ["dinov2_vitl14", "dinov2_vitb14", "resnet50", "vit_b16"]:
            print(f"\n{'#'*70}\n# BACKBONE: {bb}\n{'#'*70}")
            run_step(args.step, bb, force=args.force)
    else:
        run_step(args.step, args.backbone, force=args.force)

    print(f"\n[ALL DONE] step={args.step}  backbone={args.backbone}  force={args.force}")
