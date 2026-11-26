"""
spikein_shuffle_gradient_v2.py - 内存优化版，支持跳过已完成的强度
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
    m = len(probs)
    n_shuffle = int(m * strength)
    if n_shuffle == 0:
        return probs, argmax
    shuffle_idx = rng.choice(m, size=n_shuffle, replace=False)
    perm = rng.permutation(n_shuffle)
    probs_new = probs.copy()
    argmax_new = argmax.copy()
    probs_new[shuffle_idx] = probs[shuffle_idx][perm]
    argmax_new[shuffle_idx] = argmax[shuffle_idx][perm]
    return probs_new, argmax_new


def run_enet_one(X, y, cov, n_seeds=1):
    if np.std(y) < 1e-8:
        return np.nan, np.nan
    result = evaluate_enet_nested(X, y, cov=cov, n_seeds=n_seeds)
    r = result["pearson_mean"]
    n = len(y)
    if np.isnan(r) or abs(r) >= 1.0 or n <= 2:
        p = np.nan
    else:
        from scipy.stats import t as t_dist
        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
        p = 2 * t_dist.sf(abs(t_stat), df=n - 2)
    return r, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--strengths", type=str, default="0.5,0.75,1.0")
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

    print(f"[load] cases={len(case_ids)}, proteins={len(gt)}", flush=True)

    for s in strengths:
        out_file = out_dir / f"result_s{s:.2f}.csv"
        if out_file.exists():
            print(f"[skip] s={s:.2f} already done", flush=True)
            continue

        print(f"[s={s:.2f}] extracting features...", flush=True)
        features = []
        for cid in case_ids:
            probs = np.load(niche_dir / f"{cid}_probs.npy")
            argmax = np.load(niche_dir / f"{cid}_argmax.npy")
            if s > 0:
                probs, argmax = partial_shuffle(probs, argmax, s, rng)
            feat = extract_one(probs, argmax)
            features.append(feat)
            del probs, argmax
        gc.collect()

        X = np.stack(features)
        feat_cols = FEATURE_NAMES
        cov = clinical[["age", "sex", "subtype"]].values.astype(np.float64)
        del features
        gc.collect()

        print(f"[s={s:.2f}] running ElasticNet...", flush=True)
        results = []
        for col in prot.columns:
            if col == "case_id":
                continue
            y = prot[col].values.astype(np.float64)
            r, p = run_enet_one(X, y, cov, n_seeds=args.n_seeds)
            results.append({"protein": col, "pearson": r, "p": p})
        gc.collect()

        df = pd.DataFrame(results)
        pvals = df["p"].fillna(1.0).values
        df["q"] = bh_fdr(pvals)
        df["hit"] = df["q"] < 0.05
        df.to_csv(out_file, index=False)

        hits = set(df[df["hit"]]["protein"])
        c_hits = len(hits & c_prots)
        a_hits = len(hits & a_prots)
        n_hits = len(hits & n_prots)
        print(f"  C={c_hits}/{len(c_prots)}, A={a_hits}/{len(a_prots)}, "
              f"N={n_hits}/{len(n_prots)}", flush=True)
        del df, X
        gc.collect()

    # 汇总所有结果
    print("\n=== SUMMARY ===", flush=True)
    rows = []
    all_s = sorted([0.0, 0.25] + strengths)
    for s in all_s:
        f = out_dir / f"result_s{s:.2f}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        hits = set(df[df["hit"]]["protein"])
        rows.append({
            "strength": s,
            "c_recall": len(hits & c_prots) / len(c_prots),
            "a_recall": len(hits & a_prots) / len(a_prots),
            "m_recall": len(hits & m_prots) / len(m_prots),
            "n_fpr": len(hits & n_prots) / len(n_prots),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "shuffle_gradient.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    if len(summary) >= 2:
        a_recalls = summary["a_recall"].values
        c_recalls = summary["c_recall"].values
        a_monotonic = all(a_recalls[i] >= a_recalls[i+1] for i in range(len(a_recalls)-1))
        c_range = max(c_recalls) - min(c_recalls)
        a_drop = a_recalls[0] - a_recalls[-1]
        print(f"\nA monotonic decreasing: {a_monotonic}")
        print(f"C recall range: {c_range:.3f}")
        print(f"A total drop: {a_drop:.3f}")


if __name__ == "__main__":
    main()
