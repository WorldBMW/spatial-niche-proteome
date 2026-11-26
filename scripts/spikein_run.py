"""
spikein_run.py
==============
对 spike-in 数据集运行五轴压力测试协议，并与 ground truth 比对。

用法:
    python scripts/spikein_run.py --data data/spikein_smoke --out results/spikein_smoke
    python scripts/spikein_run.py --data data/spikein_full --out results/spikein_full

核心指标:
    1. 逐轴过滤的召回率/精度/F1 (表 1)
    2. 组成/排列分解准确率 (表 2)
    3. 与真实数据的衰减比例对照 (表 4)
"""
import sys, os, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from downstream_omics import evaluate_enet_nested, bh_fdr

SEED = 2026


def load_data(data_dir: Path):
    """加载 spike-in 数据集。"""
    clinical = pd.read_csv(data_dir / "clinical.csv")
    feat_raw = pd.read_csv(data_dir / "features_raw.csv")
    feat_shuf = pd.read_csv(data_dir / "features_shuffled.csv")
    prot = pd.read_csv(data_dir / "proteomics.csv")
    gt = pd.read_csv(data_dir / "ground_truth.csv")

    # 对齐 case_id
    case_ids = feat_raw["case_id"].tolist()
    prot = prot.set_index("case_id").loc[case_ids].reset_index()
    clinical = clinical.set_index("case_id").loc[case_ids].reset_index()

    return clinical, feat_raw, feat_shuf, prot, gt


def run_enet_one(X: np.ndarray, y: np.ndarray, cov: np.ndarray | None,
                 n_seeds: int = 1) -> dict:
    """对单个蛋白跑 ElasticNet 嵌套 CV，返回 pearson r 和 p-value。"""
    if np.std(y) < 1e-8:
        return {"pearson": np.nan, "p": np.nan, "r2": np.nan}
    result = evaluate_enet_nested(X, y, cov=cov, n_seeds=n_seeds)
    r = result["pearson_mean"]
    n = len(y)
    # 快速 p-value (t 检验, smoke test 用; 正式实验用 permutation)
    if np.isnan(r) or abs(r) >= 1.0 or n <= 2:
        p = np.nan
    else:
        from scipy.stats import t as t_dist
        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
        p = 2 * t_dist.sf(abs(t_stat), df=n - 2)
    return {"pearson": r, "p": p, "r2": result["r2_mean"]}


def run_axis(feat_df: pd.DataFrame, prot_df: pd.DataFrame,
             clinical: pd.DataFrame, cov_mode: str = "none",
             feature_subset: str = "full", n_seeds: int = 1) -> pd.DataFrame:
    """
    运行一轴，返回每个蛋白的结果 DataFrame。
    cov_mode: "none" | "age_sex" | "full" (age+sex+subtype)
    feature_subset: "full" | "only-abund" | "only-spatial"
    """
    case_ids = feat_df["case_id"].tolist()
    feat_cols = [c for c in feat_df.columns
                 if c not in ("case_id", "subtype")]

    if feature_subset == "only-abund":
        feat_cols = [c for c in feat_cols if c.startswith("abund_")]
    elif feature_subset == "only-spatial":
        feat_cols = [c for c in feat_cols if not c.startswith("abund_")]

    X = feat_df[feat_cols].values.astype(np.float64)

    # 协变量
    if cov_mode == "none":
        cov = None
    elif cov_mode == "age_sex":
        cov = clinical[["age", "sex"]].values.astype(np.float64)
    elif cov_mode == "full":
        cov = clinical[["age", "sex", "subtype"]].values.astype(np.float64)

    results = []
    for col in prot_df.columns:
        if col == "case_id":
            continue
        y = prot_df[col].values.astype(np.float64)
        r = run_enet_one(X, y, cov, n_seeds=n_seeds)
        results.append({"protein": col, **r})

    df = pd.DataFrame(results)
    # BH-FDR
    pvals = df["p"].fillna(1.0).values
    df["q"] = bh_fdr(pvals)
    df["hit"] = df["q"] < 0.05
    return df


def evaluate(results: dict, gt: pd.DataFrame) -> pd.DataFrame:
    """
    计算逐轴的召回率/精度/F1。
    results: {axis_name: result_df}
    gt: ground_truth DataFrame (protein, group, ...)
    """
    rows = []
    for axis_name, res_df in results.items():
        merged = res_df.merge(gt, on="protein", how="left")
        for group in ["C", "A", "M", "N"]:
            g = merged[merged["group"] == group]
            n_total = len(g)
            n_hits = int(g["hit"].sum())
            if n_total == 0:
                continue
            if group == "N":
                # 噪声组: 精度=0 (无真阳性), 假阳率 = n_hits / n_total
                rows.append({
                    "axis": axis_name, "group": group,
                    "n_total": n_total, "n_hits": n_hits,
                    "recall": np.nan, "precision": 0.0,
                    "f1": 0.0, "fpr": n_hits / n_total,
                })
            else:
                # 信号组: 召回率 = 命中数 / 总数, 精度 = 1.0 (因为这些都是真阳性)
                recall = n_hits / n_total
                # 精度需要跨组计算: 真阳性 / 所有命中
                all_hits = merged[merged["hit"]]
                tp = len(all_hits[all_hits["group"].isin(["C", "A", "M"])])
                fp = len(all_hits[all_hits["group"] == "N"])
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
                rows.append({
                    "axis": axis_name, "group": group,
                    "n_total": n_total, "n_hits": n_hits,
                    "recall": recall, "precision": precision,
                    "f1": f1, "fpr": np.nan,
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="spike-in 数据目录")
    ap.add_argument("--out", type=str, required=True, help="结果输出目录")
    ap.add_argument("--n-seeds", type=int, default=1, help="CV 种子数 (smoke=1, full=3)")
    args = ap.parse_args()

    data_dir = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {data_dir}", flush=True)
    clinical, feat_raw, feat_shuf, prot, gt = load_data(data_dir)
    print(f"  cases={len(clinical)}, proteins={len(gt)}, "
          f"feature_dim={len([c for c in feat_raw.columns if c not in ('case_id','subtype')])}",
          flush=True)

    # ===== 五轴协议 =====
    results = {}

    # 基线: 无协变量, 全特征
    print("[axis] baseline (no cov, full features)...", flush=True)
    results["baseline"] = run_axis(feat_raw, prot, clinical, "none", "full", args.n_seeds)

    # A2 协变量阶梯
    print("[axis] A2 age+sex...", flush=True)
    results["A2_agesex"] = run_axis(feat_raw, prot, clinical, "age_sex", "full", args.n_seeds)
    print("[axis] A2 full (age+sex+subtype)...", flush=True)
    results["A2_full"] = run_axis(feat_raw, prot, clinical, "full", "full", args.n_seeds)

    # A4 特征类型
    print("[axis] A4 abundance-only...", flush=True)
    results["A4_abund"] = run_axis(feat_raw, prot, clinical, "none", "only-abund", args.n_seeds)
    print("[axis] A4 spatial-only...", flush=True)
    results["A4_spatial"] = run_axis(feat_raw, prot, clinical, "none", "only-spatial", args.n_seeds)

    # A5 双空对照: 空间置换特征 (叠加全协变量校正, 去掉亚型后门)
    print("[axis] A5 spatial shuffle + full cov...", flush=True)
    results["A5_shuffle"] = run_axis(feat_shuf, prot, clinical, "full", "full", args.n_seeds)

    # ===== 评估 =====
    print("[eval] computing recall/precision/F1...", flush=True)
    eval_df = evaluate(results, gt)

    # 保存 (使用绝对路径 + 逐个 try-except)
    out_dir_abs = out_dir.resolve()
    eval_df.to_csv(out_dir_abs / "axis_metrics.csv", index=False)
    save_order = ["baseline", "A2_agesex", "A2_full", "A4_abund", "A4_spatial", "A5_shuffle"]
    for i, name in enumerate(save_order):
        if name not in results:
            continue
        fpath = out_dir_abs / f"result_{i:02d}_{name}.csv"
        try:
            results[name].to_csv(fpath, index=False)
            print(f"[save] {fpath.name}", flush=True)
        except Exception as e:
            print(f"[warn] skip {name}: {e}", flush=True)

    # ===== 打印摘要 =====
    print("\n" + "=" * 80)
    print("SPIKE-IN RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'axis':<20} {'group':<6} {'n_total':<8} {'n_hits':<8} "
          f"{'recall':<8} {'precision':<10} {'f1':<8} {'fpr':<8}")
    print("-" * 80)
    for _, r in eval_df.iterrows():
        rec = f"{r['recall']:.3f}" if not np.isnan(r['recall']) else "  -   "
        pre = f"{r['precision']:.3f}"
        f1 = f"{r['f1']:.3f}"
        fpr = f"{r['fpr']:.3f}" if not np.isnan(r['fpr']) else "  -   "
        print(f"{r['axis']:<20} {r['group']:<6} {r['n_total']:<8} {r['n_hits']:<8} "
              f"{rec:<8} {pre:<10} {f1:<8} {fpr:<8}")

    # 关键判据检查
    print("\n" + "=" * 80)
    print("PRE-REGISTERED CRITERIA CHECK")
    print("=" * 80)

    # 判据 1: 精度单调上升 (baseline → A2_full)
    base_prec = eval_df[(eval_df["axis"] == "baseline") & (eval_df["group"] == "C")]["precision"].values[0]
    full_prec = eval_df[(eval_df["axis"] == "A2_full") & (eval_df["group"] == "C")]["precision"].values[0]
    print(f"\n[1] Precision monotonic increase: baseline={base_prec:.3f} → A2_full={full_prec:.3f}")
    print(f"    PASS: {full_prec >= base_prec}")

    # 判据 2: C 组召回率 >= 60% (A2_full)
    c_recall = eval_df[(eval_df["axis"] == "A2_full") & (eval_df["group"] == "C")]["recall"].values[0]
    print(f"\n[2] C-group recall (A2_full) >= 0.60: {c_recall:.3f}")
    print(f"    PASS: {c_recall >= 0.60}")

    # 判据 3: A 组置换后召回率下降幅度 > C 组下降幅度 (C-A 差异, 非 A 绝对降幅)
    # 高统计功效下强信号会压低 A 组绝对降幅, 故以 C-A 差异为判据 (与手稿 §2.8 一致)
    def _recall(axis, group):
        return float(eval_df[(eval_df["axis"] == axis) & (eval_df["group"] == group)]["recall"].values[0])
    a_drop = _recall("baseline", "A") - _recall("A5_shuffle", "A")
    c_drop = _recall("baseline", "C") - _recall("A5_shuffle", "C")
    print(f"\n[3] A-group recall drop ({a_drop:.3f}) > C-group drop ({c_drop:.3f}): "
          f"A {_recall('baseline','A'):.3f}->{_recall('A5_shuffle','A'):.3f}, "
          f"C {_recall('baseline','C'):.3f}->{_recall('A5_shuffle','C'):.3f}")
    print(f"    PASS: {a_drop > c_drop}")

    # 判据 4: N 组假阳率 <= 5% (A2_full)
    n_fpr = eval_df[(eval_df["axis"] == "A2_full") & (eval_df["group"] == "N")]["fpr"].values[0]
    print(f"\n[4] N-group FPR (A2_full) <= 0.05: {n_fpr:.3f}")
    print(f"    PASS: {n_fpr <= 0.05}")

    # 判据 5: 分解准确率 (C 在 baseline 中被判为组成型的比例)
    # 组成型 = 在 A4_abund 中命中, 排列型 = 在 A4_spatial 中命中
    c_abund_hits = set(results["A4_abund"][results["A4_abund"]["hit"]]["protein"])
    a_spatial_hits = set(results["A4_spatial"][results["A4_spatial"]["hit"]]["protein"])
    c_prots = set(gt[gt["group"] == "C"]["protein"])
    a_prots = set(gt[gt["group"] == "A"]["protein"])
    c_correct = len(c_prots & c_abund_hits)
    a_correct = len(a_prots & a_spatial_hits)
    decomp_acc = (c_correct + a_correct) / (len(c_prots) + len(a_prots)) if (len(c_prots) + len(a_prots)) > 0 else 0.0
    print(f"\n[5] Decomposition accuracy >= 0.70: {decomp_acc:.3f}")
    print(f"    C-correct={c_correct}/{len(c_prots)}, A-correct={a_correct}/{len(a_prots)}")
    print(f"    PASS: {decomp_acc >= 0.70}")

    print(f"\n[done] results saved to {out_dir}")


if __name__ == "__main__":
    main()
