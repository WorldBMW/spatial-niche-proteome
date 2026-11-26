"""P1: 空间可视化 —— niche 地图 + 关键拓扑转移高亮。

用法:
  python src/visualize_trans.py --tag tcgaemb_ucec --niche-tag tcga_ucec \
      --features data/features/dinov2_vitl14_features_tcga_ucec/all_features.csv \
      --trans 0_5 0_1 4_2 --n-cases 2 --out-dir results_vitl14_jbi_pancancer/p1_vis
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_NICHES = 10
NICHE_NAMES = ["tumor_core", "macrophage", "b_cell", "fibroblast_CAF",
               "neutrophil_inflam", "t_cell_immune", "tumor_immune_interface",
               "stroma", "necrosis", "vessel"]
COLORS = plt.get_cmap("tab10").colors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="embeddings 目录后缀, 如 tcgaemb_ucec")
    ap.add_argument("--niche-tag", required=True, help="niches 目录后缀, 如 tcga_ucec")
    ap.add_argument("--features", required=True)
    ap.add_argument("--trans", nargs="+", default=["0_5", "0_1", "4_2"],
                    help="要高亮的转移对 (k_l)")
    ap.add_argument("--n-cases", type=int, default=2, help="每转移选 top/bottom 各多少 case")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    emb_dir = Path("data/processed/dinov2_vitl14_embeddings_" + args.tag)
    niche_dir = Path("data/processed/dinov2_vitl14_niches_" + args.niche_tag)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(args.features)
    if "case_id" in feat.columns:
        feat = feat.set_index("case_id")

    for tr in args.trans:
        k, l = (int(v) for v in tr.split("_"))
        col = f"trans_{k}_{l}"
        if col not in feat.columns:
            print(f"[skip] {col} 不在 features")
            continue
        vals = feat[col].dropna()
        picks = list(vals.nlargest(args.n_cases).index) + list(vals.nsmallest(args.n_cases).index)
        print(f"\n=== trans {col}: 高值 case {list(vals.nlargest(args.n_cases).index)}, "
              f"低值 case {list(vals.nsmallest(args.n_cases).index)} ===")

        for rank, case_id in enumerate(picks):
            hi_low = "high" if rank < args.n_cases else "low"
            fp = niche_dir / f"{case_id}_argmax.npy"
            cp = emb_dir / f"{case_id}_coords.csv"
            if not fp.exists() or not cp.exists():
                print(f"[skip] {case_id} 缺 argmax/coords")
                continue
            argmax = np.load(fp)
            coords = pd.read_csv(cp)[["x", "y"]].values
            n = min(len(argmax), len(coords))
            argmax, coords = argmax[:n], coords[:n]

            fig, axes = plt.subplots(1, 2, figsize=(17, 8))
            # 左: niche 地图
            ax = axes[0]
            for nic in range(N_NICHES):
                m = argmax == nic
                if m.sum() == 0:
                    continue
                ax.scatter(coords[m, 0], coords[m, 1], s=5, c=[COLORS[nic]],
                           label=f"{nic} {NICHE_NAMES[nic]} ({m.sum()})", alpha=0.7,
                           linewidths=0)
            ax.set_title(f"{case_id} niche map\ntrans_{k}_{l}={vals[case_id]:.3f} ({hi_low})")
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.legend(fontsize=6, markerscale=3, loc="best")

            # 右: 转移高亮 (光栅相邻对 k→l)
            ax = axes[1]
            for nic in range(N_NICHES):
                m = argmax == nic
                if m.sum() == 0:
                    continue
                ax.scatter(coords[m, 0], coords[m, 1], s=4, c=[COLORS[nic]],
                           alpha=0.25, linewidths=0)
            seg = 0
            for i in range(n - 1):
                if argmax[i] == k and argmax[i + 1] == l:
                    xs = [coords[i, 0], coords[i + 1, 0]]
                    ys = [coords[i, 1], coords[i + 1, 1]]
                    ax.plot(xs, ys, "-", c="red", lw=1.2, alpha=0.8)
                    seg += 1
            ax.set_title(f"{k} {NICHE_NAMES[k]} -> {l} {NICHE_NAMES[l]}\n"
                         f"{seg} raster-adjacent transitions")
            ax.invert_yaxis()
            ax.set_aspect("equal")

            out = out_dir / f"{args.niche_tag}_{case_id}_trans_{k}_{l}_{hi_low}.png"
            fig.savefig(out, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  [save] {out}")

if __name__ == "__main__":
    main()
