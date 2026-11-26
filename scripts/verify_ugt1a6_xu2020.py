# -*- coding: utf-8 -*-
"""
verify_ugt1a6_xu2020.py — 在独立 LUAD 蛋白组队列 (Xu et al. Cell 2020, n=103) 复核 UGT1A6 方向

预设定检验 (primary):
  Spearman r(UGT1A6 蛋白丰度, 肺上皮分化模块) 在 103 例肿瘤中 > 0 ?
  -- 与本文 UGT1A6 的正向 (r=+0.332) 空间关联方向对照

次设检验 (secondary):
  UGT1A6 蛋白丰度在已发表蛋白组亚型 (S-I/S-II/S-III) 间是否不同 (Kruskal-Wallis)
  以及与间质模块的相关 (排除"仅是间质混入")

输出: results_dict_encoder_sensitivity/ugt1a6_xu2020_direction.csv + 控制台报告
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "xu2020_luad"
OUT = ROOT / "results_dict_encoder_sensitivity" / "ugt1a6_xu2020_direction.csv"

TARGET = "UGT1A6"

EPI_MODULE = ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "MUC1", "CLDN18",
              "SFTPB", "SFTPC", "NAPSA", "NKX2-1", "TTF1"]
STROMA_MODULE = ["VIM", "COL1A1", "COL1A2", "ACTA2", "DCN", "FN1"]
WATCH = [TARGET, "SLPI", "WNT2", "MUC5AC", "MUC5B"]

rng = np.random.default_rng(42)
N_PERM = 10000


def to_num(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in ("", "NA", "NaN", "nan", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------- 1. 读 clinical (Table S1A) ----------
wb = load_workbook(EXT / "mmc1.xlsx", read_only=True, data_only=True)
ws = wb["Table S1A"]
rows = list(ws.iter_rows(values_only=True))
wb.close()

header = [str(h) if h is not None else "" for h in rows[0]]
print(f"Table S1A: {len(rows)-1} 患者, {len(header)} 列")
id_col = next(i for i, h in enumerate(header) if "Sample ID" in h)
diff_col = next(i for i, h in enumerate(header) if "Degree of differentiation" in h)
pur_col = next(i for i, h in enumerate(header) if "Tumor purity" in h)
stroma_col = next(i for i, h in enumerate(header) if h.strip().startswith("Stroma"))

clinical = {}
for r in rows[1:]:
    if not r or r[id_col] is None:
        continue
    sid = str(r[id_col]).strip()
    clinical[sid] = dict(
        diff=to_num(r[diff_col]), purity=to_num(r[pur_col]), stroma=to_num(r[stroma_col]))
print(f"临床样本数: {len(clinical)} | 分化/纯度/间质列: "
      f"{header[diff_col]} / {header[pur_col]} / {header[stroma_col]}")

# ---------- 2. 流式读 Table S4B, 只保留目标基因 ----------
wb = load_workbook(EXT / "mmc4.xlsx", read_only=True, data_only=True)
ws = wb["Table S4B"]
it = ws.iter_rows(values_only=True)
hdr = list(next(it))
gene_col = 0

# 列名形如 "iBAQ LUAD_01T" / "..._01N"
tum_cols = [i for i, h in enumerate(hdr)
            if h and re.search(r"LUAD_\d+T\b", str(h))]
print(f"Table S4B: {len(hdr)} 列, 肿瘤列 {len(tum_cols)}")

want = set(EPI_MODULE + STROMA_MODULE + WATCH)
gene_rows = {}
for r in it:
    if not r or r[gene_col] is None:
        continue
    g = str(r[gene_col]).strip()
    if g in want and g not in gene_rows:
        gene_rows[g] = r
wb.close()

print("命中基因:", sorted(gene_rows))

samples = [re.search(r"(LUAD_\d+)", str(hdr[i])).group(1) for i in tum_cols]

def vec(g):
    r = gene_rows[g]
    return np.array([to_num(r[i]) for i in tum_cols], dtype=float)

# ---------- 3. 构建模块分数 ----------
def module_score(genes):
    used, zs = [], []
    for g in genes:
        if g not in gene_rows:
            continue
        v = vec(g)
        if np.isfinite(v).sum() < 30:
            continue
        lg = np.log2(v + 1.0)
        mu, sd = np.nanmean(lg), np.nanstd(lg)
        if not np.isfinite(sd) or sd == 0:
            continue
        zs.append((lg - mu) / sd)
        used.append(g)
    if len(used) < 4:
        return None, used, None
    M = np.vstack(zs)
    score = np.nanmean(M, axis=0)
    return score, used, M

epi, epi_used, _ = module_score(EPI_MODULE)
stro, stro_used, _ = module_score(STROMA_MODULE)
print(f"上皮模块 ({len(epi_used)}): {epi_used}")
print(f"间质模块 ({len(stro_used)}): {stro_used}")

u = vec(TARGET)
log_u = np.log2(u + 1.0)

def perm_p(x, y, n_perm=N_PERM):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20:
        return np.nan, np.nan, len(x)
    r = stats.spearmanr(x, y).statistic
    cnt = 0
    for _ in range(n_perm):
        if stats.spearmanr(x, rng.permutation(y)).statistic >= r:
            cnt += 1
    return r, (cnt + 1) / (n_perm + 1), len(x)

results = []
# primary
r, p, n = perm_p(log_u, epi)
results.append(dict(test="primary: UGT1A6 vs epithelial module",
                    r=r, p=p, p_kind="permutation", n=n))
print(f"\n[PRIMARY] UGT1A6 vs 上皮分化模块: rho={r:.4f}, p_perm={p:.4g}, n={n}")

# secondary: stroma
r2, p2, n2 = perm_p(log_u, stro)
results.append(dict(test="control: UGT1A6 vs stromal module", r=r2, p=p2,
                    p_kind="permutation", n=n2))
print(f"[CONTROL] UGT1A6 vs 间质模块: rho={r2:.4f}, p_perm={p2:.4g}, n={n2}")

# per-gene correlations (成对完整样本)
print("\n逐基因 Spearman (log2 iBAQ, 成对完整样本):")
for g in epi_used + [x for x in STROMA_MODULE if x in gene_rows] + ["SLPI", "MUC5AC"]:
    if g in gene_rows:
        lg = np.log2(vec(g) + 1.0)
        ok = np.isfinite(log_u) & np.isfinite(lg)
        if ok.sum() < 10:
            print(f"  {TARGET} vs {g:8s}: 有效样本不足 (n={int(ok.sum())})")
            continue
        rg = stats.spearmanr(log_u[ok], lg[ok])
        print(f"  {TARGET} vs {g:8s}: rho={rg.statistic:+.3f} (p={rg.pvalue:.3g}, n={int(ok.sum())})")
        results.append(dict(test=f"per-gene: {TARGET} vs {g}", r=rg.statistic,
                            p=rg.pvalue, p_kind="analytic", n=int(ok.sum())))

# secondary: 病理学对应指标 (分化程度 / 肿瘤纯度 / 间质比例)
print("\n[CLIN] UGT1A6 与病理指标 (仅在检出的样本内):")
detected = int(np.isfinite(log_u).sum())
print(f"       检出率: {detected}/{len(log_u)} 例")
for key, label in [("diff", "Degree of differentiation (1 poor - 3 well)"),
                   ("purity", "Tumor purity (%)"),
                   ("stroma", "Stroma (%)")]:
    vals = np.array([clinical.get(s, {}).get(key, np.nan) for s in samples], dtype=float)
    ok = np.isfinite(log_u) & np.isfinite(vals)
    if ok.sum() >= 12:
        rho, pv = stats.spearmanr(log_u[ok], vals[ok])
        print(f"        vs {label}: rho={rho:+.3f}, p={pv:.4g}, n={int(ok.sum())}")
        results.append(dict(test=f"clinical: {TARGET} vs {label}",
                            r=rho, p=pv, p_kind="analytic", n=int(ok.sum())))
    else:
        print(f"        vs {label}: 有效样本不足 (n={int(ok.sum())})")

df = pd.DataFrame(results)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"\n结果保存: {OUT}")

# 关键数字摘要
print("\n" + "=" * 60)
print(f"UGT1A6 在 103 例 LUAD 肿瘤中的检出: "
      f"{int(np.isfinite(u).sum())}/103, 中位 log2 iBAQ={np.nanmedian(log_u):.2f}")
print(f"方向判定: 上皮模块 rho={r:+.3f} -> "
      f"{'与本文正向关联方向一致' if r > 0 else '方向相反'}")
