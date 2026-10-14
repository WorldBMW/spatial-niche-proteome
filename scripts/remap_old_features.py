"""
scripts/remap_old_features.py
============================
如果服务器上已经有 features CSV 但 case_id 是旧的 (TCGA-S0000, CPTAC-S0000)，
用此脚本清理后重跑。

**最简单的做法**: 直接清掉旧的中间文件，重跑 pipeline 的后半段。

用法：
  python scripts/remap_old_features.py --backbone dinov2_vitl14 --action clean
  python scripts/remap_old_features.py --backbone dinov2_vitl14 --action remap  # 尝试修复
"""
import sys, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import config
from case_id import resolve_case_id, is_uuid


def find_bad_files(backbone: str):
    """找出所有 case_id 有问题的中间文件。"""
    bad = []

    # 1. embedding 目录 — 检查 .npy 文件名
    embed_dir = config.embed_cache_dir(backbone)
    if embed_dir.exists():
        for f in sorted(embed_dir.glob("*.npy")):
            if "_coords" in f.name:
                continue
            if is_uuid(f.stem) or "S0" in f.stem:
                bad.append(("embedding", f))

    # 2. niche 目录
    niche_dir = config.niche_cache_dir(backbone)
    if niche_dir.exists():
        for f in sorted(niche_dir.glob("*_probs.npy")):
            case_id = f.name.replace("_probs.npy", "")
            if is_uuid(case_id) or "S0" in case_id:
                bad.append(("niche", f))

    # 3. features CSV
    feat_csv = config.features_dir(backbone) / "all_features.csv"
    if feat_csv.exists():
        try:
            df = pd.read_csv(feat_csv)
            sample_ids = df["case_id"].head(3).tolist()
            if any("S0" in str(c) or is_uuid(str(c)) for c in sample_ids):
                bad.append(("features", feat_csv))
        except Exception:
            pass

    return bad


def clean(backbone: str, dry_run: bool = False):
    """删除所有有问题的中间文件。"""
    bad = find_bad_files(backbone)
    if not bad:
        print(f"[clean] {backbone}: 没发现有问题的文件 ✓")
        return

    print(f"[clean] {backbone}: 发现 {len(bad)} 个有问题的文件:")
    for category, path in bad:
        flag = "[DRY-RUN] " if dry_run else ""
        print(f"  {flag}{category}: {path}")
        if not dry_run:
            path.unlink(missing_ok=True)
            print(f"    → DELETED")

    # 也删 _coords.csv 和 _argmax.npy / _probs.npy 的对应文件
    if not dry_run:
        for category, path in bad:
            if path.suffix == ".npy" and "_coords" not in path.name:
                # 删掉对应的 coords.csv
                coords = path.parent / f"{path.stem}_coords.csv"
                coords.unlink(missing_ok=True)
                # 删掉 niche 输出
                for suffix in ["_probs.npy", "_argmax.npy"]:
                    niche_path = config.niche_cache_dir(backbone) / f"{path.stem}{suffix}"
                    niche_path.unlink(missing_ok=True)

    print(f"\n[clean] 完成。现在可以重跑:")
    print(f"  python src/pipeline.py --step all --backbone {backbone}")
    print(f"  (embed 会自动用 resolve_case_id 解析真实 TCGA barcode)")


def remap_features(backbone: str):
    """尝试修复 features CSV 的 case_id（只有当有 UUID→barcode 映射时才有用）。"""
    feat_csv = config.features_dir(backbone) / "all_features.csv"
    if not feat_csv.exists():
        print("[remap] features CSV 不存在")
        return

    df = pd.read_csv(feat_csv)
    print(f"[remap] 原始 case_ids sample: {df['case_id'].head(5).tolist()}")

    # 尝试从 embedding 目录找 UUID → barcode 映射
    embed_dir = config.embed_cache_dir(backbone)
    uuid_to_barcode = {}
    if embed_dir.exists():
        for f in sorted(embed_dir.glob("*.npy")):
            if "_coords" in f.name:
                continue
            stem = f.stem
            if is_uuid(stem):
                # UUID 文件先不删，重跑 embed 会自动覆盖
                pass
            else:
                resolved = resolve_case_id(stem)
                # 这个 resolved 就是真实 barcode

    # 如果是 S0000 格式，只能用手工映射或重跑
    if df["case_id"].str.contains("S0").any():
        print("[remap] features CSV 含 TCGA-S0000/CPTAC-S0000 格式。")
        print("  这种格式无法自动映射 — 请用 --action clean 清掉后重跑。")
        return

    print(f"[remap] 重跑 embed/niche/features 来重新生成...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitl14")
    ap.add_argument("--action", choices=["clean", "remap", "check"], default="check")
    ap.add_argument("--dry-run", action="store_true", help="不真删，只预览")
    args = ap.parse_args()

    if args.action == "check":
        bad = find_bad_files(args.backbone)
        if bad:
            print(f"[{args.backbone}] 发现 {len(bad)} 个有问题的文件:")
            for cat, p in bad:
                print(f"  {cat}: {p}")
            print("\n建议: python scripts/remap_old_features.py --action clean")
        else:
            print(f"[{args.backbone}] 一切正常 ✓")
    elif args.action == "clean":
        clean(args.backbone, dry_run=args.dry_run)
    elif args.action == "remap":
        remap_features(args.backbone)
