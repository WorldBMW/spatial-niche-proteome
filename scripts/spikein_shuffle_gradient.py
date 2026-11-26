"""
spikein_shuffle_gradient.py
============================
置换强度梯度实验：验证空间置换对排列型(A)信号的特异性破坏随强度单调增强，
而组成型(C)信号不受影响。

置换强度 s ∈ {0, 0.25, 0.5, 0.75, 1.0}：
  - 对每例 WSI，随机选择 s 比例的 tile 进行行置换（打乱其 niche 分配）
  - s=0: 原始特征（不置换）
  - s=1.0: 全量行置换（当前 A5 对照）

用法:
    python scripts/spikein_shuffle_gradient.py --data data/spikein_full --out results/spikein_shuffle_gradient
"""
import sys, argparse, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import extract_one, FEATURE_NAMES
from downstream_omics import evaluate_enet_nested, bh_fdr


def partial_shuffle(probs, argmax, strength, rng):
    """对 strength 比例的 tile 做行置换，其余保持原样。"""
    m = len(probs)
    n_shuffle = int(m * strength)
    if n_shuffle == 0:
        return probs.copy(), argmax.copy()
    # 随机选择要置换的 tile 索引
    shuffle_idx = rng.choice(m, size=n_shuffle, replace=False)
    # 对这些索引做置换
    perm = rng.permutation(n_shuffle)
    probs_new = probs.copy()
    argmax_new = argmax.copy()
    probs_new[shuffle_idx] = probs[shuffle_idx][perm]
    argmax_new[shuffle_idx] = argmax[shuffle_idx][perm]
    return probs_new, argmax_new


def run_enet_one(X, y, cov, n_seeds=1):
    if np.std(y) < 1e-8:
        return {"pearson": np.nan, "p": np.nan, "r2": np.nan}
    result = evaluate_enet_nested(X, y, cov=cov, n_seeds=n_seeds)
    r = result["pearson_mean"]
    n = len(y)
    if np.isnan(r) or abs(r) >= 1.0 or n <= 2:
        p = np.nan
    else:
        from scipy.stats import t as t_dist
        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
        p = 2 * t_dist.sf(abs(t_stat), df=n - 2)
    return {"pearson": r, "p": p, "r2": result["r2_mean"]}


def run_axis(feat_df, prot_df, clinical, n_seeds=1):
    """全特征 + 全协变量校正（A2_full 配置）。"""
    case_ids = feat_df["case_id"].tolist()
    feat_cols = [c for c in feat_df.columns if c not in ("case_id", "subtype")]
    X = feat_df[feat_cols].values.astype(np.float64)
    cov = clinical[["age", "sex", "subtype"]].values.astype(np.float64)
    results = []
    for col in prot_df.columns:
        if col == "case_id":
            continue
        y = prot_df[col].values.astype(np.float64)
        r = run_enet_one(X, y, cov, n_seeds=n_seeds)
        results.append({"protein": col, **r})
    df = pd.DataFrame(results)
    pvals = df["p"].fillna(1.0).values
    df["q"] = bh_fdr(pvals)
    df["hit"] = df["q"] < 0.05
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="spike-in 数据目录")
    ap.add_argument("--out", type=str, required=True, help="结果输出目录")
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--strengths", type=str, default="0,0.25,0.5,0.75,1.0",
                    help="置换强度梯度，逗号分隔")
    args = ap.parse_args()

    data_dir = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    niche_dir = data_dir / "niche_maps"

    strengths = [float(s) for s in args.strengths.split(",")]
    rng = np.random.RandomState(2026)

    clinical = pd.read_csv(data_dir / "clinical.csv")
    prot = pd.read_csv(data_dir / "proteomics.csv")
    gt = pd.read_csv(data_dir / "ground_truth.csv")
    case_ids = clinical["case_id"].tolist()
    prot = prot.set_index("case_id").loc[case_ids].reset_index()

    c_prots = set(gt[gt["group"] == "C"]["protein"])
    a_prots = set(gt[gt["group"] == "A"]["protein"])
    m_prots = set(gt[gt["group"] == "M"]["protein"])
    n_prots = set(gt[gt["group"] == "N"]["protein"])

    print(f"[load] {data_dir}", flush=True)
    print(f"  cases={len(case_ids)}, proteins={len(gt)}, strengths={strengths}", flush=True)

    # 预加载所有 niche 地图
    print("[load] reading niche maps...", flush=True)
    all_probs = {}
    all_argmax = {}
    for cid in case_ids:
        all_probs[cid] = np.load(niche_dir / f"{cid}_probs.npy")
        all_argmax[cid] = np.load(niche_dir / f"{cid}_argmax.npy")
    print(f"  loaded {len(all_probs)} niche maps", flush=True)

    gradient_results = []
    for s in strengths:
        print(f"\n[strength={s:.2f}] extracting features...", flush=True)
        features = []
        for cid in case_ids:
            probs = all_probs[cid]
            argmax = all_argmax[cid]
            if s > 0:
                probs_s, argmax_s = partial_shuffle(probs, argmax, s, rng)
            else:
                probs_s, argmax_s = probs, argmax
            feat = extract_one(probs_s, argmax_s)
            features.append(feat)
        X = np.stack(features)
        feat_df = pd.DataFrame(X, columns=FEATURE_NAMES)
        feat_df.insert(0, "case_id", case_ids)
        feat_df["subtype"] = clinical["subtype"].values

        print(f"[strength={s:.2f}] running ElasticNet (full cov)...", flush=True)
        res_df = run_axis(feat_df, prot, clinical, n_seeds=args.n_seeds)

        c_hits = len(set(res_df[res_df["hit"]]["protein"]) & c_prots)
        a_hits = len(set(res_df[res_df["hit"]]["protein"]) & a_prots)
        m_hits = len(set(res_df[res_df["hit"]]["protein"]) & m_prots)
        n_hits = len(set(res_df[res_df["hit"]]["protein"]) & n_prots)

        c_recall = c_hits / len(c_prots)
        a_recall = a_hits / len(a_prots)
        m_recall = m_hits / len(m_prots)
        n_fpr = n_hits / len(n_prots)

        row = {
            "strength": s,
            "c_hits": c_hits, "c_total": len(c_prots), "c_recall": c_recall,
            "a_hits": a_hits, "a_total": len(a_prots), "a_recall": a_recall,
            "m_hits": m_hits, "m_total": len(m_prots), "m_recall": m_recall,
            "n_hits": n_hits, "n_total": len(n_prots), "n_fpr": n_fpr,
        }
        gradient_results.append(row)
        print(f"  C recall={c_recall:.3f} ({c_hits}/{len(c_prots)}), "
              f"A recall={a_recall:.3f} ({a_hits}/{len(a_prots)}), "
              f"N FPR={n_fpr:.3f} ({n_hits}/{len(n_prots)})", flush=True)

        # 保存该强度的详细结果
        res_df.to_csv(out_dir / f"result_s{s:.2f}.csv", index=False)
        gc.collect()

    # 汇总
    grad_df = pd.DataFrame(gradient_results)
    grad_df.to_csv(out_dir / "shuffle_gradient.csv", index=False)

    print("\n" + "=" * 70)
    print("SHUFFLE STRENGTH GRADIENT RESULTS")
    print("=" * 70)
    print(f"{'strength':<10} {'C_recall':<10} {'A_recall':<10} {'A_drop':<10} "
          f"{'C_A_gap':<10} {'N_FPR':<10}")
    print("-" * 70)
    base_a = grad_df.iloc[0]["a_recall"]
    for _, r in grad_df.iterrows():
        a_drop = base_a - r["a_recall"]
        gap = r["c_recall"] - r["a_recall"]
        print(f"{r['strength']:<10.2f} {r['c_recall']:<10.3f} {r['a_recall']:<10.3f} "
              f"{a_drop:<10.3f} {gap:<10.3f} {r['n_fpr']:<10.3f}")

    # 单调性检验
    a_recalls = grad_df["a_recall"].values
    c_recalls = grad_df["c_recall"].values
    a_monotonic = all(a_recalls[i] >= a_recalls[i+1] for i in range(len(a_recalls)-1))
    c_stable = max(c_recalls) - min(c_recalls) < 0.2
    print(f"\n[check] A recall monotonic decreasing: {a_monotonic}")
    print(f"[check] C recall stable (range < 0.2): {c_stable} (range={max(c_recalls)-min(c_recalls):.3f})")
    print(f"[check] A drop at s=1.0 >= 0.5: {base_a - a_recalls[-1] >= 0.5} "
          f"(drop={base_a - a_recalls[-1]:.3f})")
    print(f"\n[done] results saved to {out_dir}")


if __name__ == "__main__":
    main()
