"""
summary.py
==========
多 backbone 结果汇总 — 把各 backbone 的 downstream_summary.csv 合并成一张对比表。

用法：
  python src/summary.py                                    # 自动跑所有 backbone
  python src/summary.py --backbones dinov2_vitl14 resnet50
  python src/summary.py --out results/comparison_summary.csv
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config


def load_backbone_results(backbone: str) -> pd.DataFrame | None:
    fp = config.RESULTS / backbone / "downstream_summary.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        df["backbone"] = backbone
        return df
    return None


def run(backbones: list[str] | None = None, out_path: Path | None = None):
    backbones = backbones or list(config.BACKBONES.keys())
    out_path = Path(out_path) if out_path else config.RESULTS / "comparison_summary.csv"

    all_frames = []
    for bb in backbones:
        df = load_backbone_results(bb)
        if df is not None:
            all_frames.append(df)
        else:
            print(f"[summary] skip {bb}: results/{bb}/downstream_summary.csv not found")

    if not all_frames:
        print("[summary] 没有任何 backbone 结果可以汇总")
        return

    merged = pd.concat(all_frames, ignore_index=True)

    # --- 生成对比表: 按 task × cohort 透视 backbone ---
    # auc_mean 作为主指标
    try:
        pivot = merged.pivot_table(
            index=["task", "cohort"], columns="backbone",
            values="auc_mean", aggfunc="mean"
        )
        # 加 C-index
        pivot_cindex = merged.pivot_table(
            index=["task", "cohort"], columns="backbone",
            values="cindex_mean", aggfunc="mean"
        )
    except Exception:
        pivot = None

    merged.to_csv(out_path, index=False)
    print(f"[summary] 全量明细 → {out_path}")

    if pivot is not None and not pivot.empty:
        # 加均值/最大值列
        numeric_cols = pivot.select_dtypes(include=[np.number]).columns
        pivot["mean"] = pivot[numeric_cols].mean(axis=1)
        pivot["best"] = pivot[numeric_cols].max(axis=1)
        pivot_out = out_path.parent / "comparison_pivot.csv"
        pivot.to_csv(pivot_out)
        print(f"[summary] 透视表 → {pivot_out}")
        print("\n=== Backbone 对比 (AUC/C-index per task) ===")
        print(pivot.to_string())
    else:
        # 简单打印
        cols = [c for c in ["task", "backbone", "cohort", "auc_mean", "auc_std",
                            "cindex_mean", "n", "p_value_holm"]
                if c in merged.columns]
        print("\n=== Results ===")
        print(merged[cols].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.backbones, args.out)
