# -*- coding: utf-8 -*-
"""生成图形摘要（Graphical Abstract）：五轴形态—分子关联压力测试协议流程图。

版式约束：
- 画布 133 x 74 数据单位，输出 figure 13.3 x 7.4 inch @200 dpi（2660 x 1480 px）；
- 三栏：左（输入）、中（五轴 + 输出三分诊）、右（六类静默故障检查表）；
- 底部横条放关键数字，逐行分行避免溢出；
- 所有文字行长度按 box 宽度反推，禁止跨框溢出。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"G:\ai_paper\experiment2_fm_time\papers\BIB_Original\submission_figures_final\Fig0_graphical_abstract.png"
CANVAS_W, CANVAS_H = 2948, 1179  # 投稿画布尺寸，与原 Fig0 一致；渲染后居中贴到白底画布上

W, H = 133.0, 74.0
fig, ax = plt.subplots(figsize=(13.3, 7.4), dpi=200)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=9.0, bold=False, ha="center", lsp=1.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.4",
            linewidth=1.1, facecolor=fc, edgecolor=ec,
        )
    )
    tx = x + w / 2 if ha == "center" else x + 1.6
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=lsp)


def arrow(x1, y1, x2, y2, color="#37474F", lw=1.2):
    ax.add_patch(
        FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                        linewidth=lw, color=color, shrinkA=0, shrinkB=0)
    )


# ---------------- 标题 ----------------
ax.text(W / 2, 70.6, "五轴形态—分子关联压力测试协议",
        ha="center", va="center", fontsize=17, fontweight="bold", color="#1A237E")
ax.text(W / 2, 67.4, "A five-axis stress-testing protocol for pathology-FM to bulk-omics associations",
        ha="center", va="center", fontsize=10, color="#455A64", style="italic")
ax.text(W / 2, 64.6, "把“关联是否可信”从作者自述，转为可复核、可套用的操作序列",
        ha="center", va="center", fontsize=10.5, color="#455A64")

# ---------------- 左：输入 ----------------
box(2.0, 32.0, 18.0, 15.0,
    "输入\n\nH&E 全切片（WSI）\n+\n同一患者 bulk 组学\n（蛋白 / 转录组）",
    "#E3F2FD", "#1565C0", 9.6, lsp=1.45)

# ---------------- 中：五轴（自上而下 A1→A5） ----------------
axes_txt = [
    ("A1  统计协议", "折级汇总 vs 合并折外\n→ 推断是否合法"),
    ("A2  协变量阶梯", "年龄/性别 → 亚型 → 组成分数\n→ 组成混杂解释多少"),
    ("A3  字典来源", "转导式 vs 冻结式\n→ 预处理依赖度"),
    ("A4  特征类型", "空间 230-d vs 池化 1024-d\n→ 排列信息增量"),
    ("A5  双空对照", "患者置换 + 片内空间置换\n→ 组成 / 排列定量分离"),
]
AX, AW, AH, AGAP = 24.0, 30.0, 7.4, 2.0
AX_TOP = 61.5
centers = []
for i, (name, desc) in enumerate(axes_txt):
    y = AX_TOP - AH - i * (AH + AGAP)
    box(AX, y, AW, AH, f"{name}\n{desc}", "#FFF8E1", "#EF6C00", 9.2, lsp=1.42)
    centers.append(y + AH / 2)
AX_R = AX + AW
AX_MID = sum(centers) / len(centers)

# 输入 → 五轴
arrow(20.4, 39.5, AX - 0.4, AX_MID)

# ---------------- 中右：三分诊输出 ----------------
outs = [
    ("排列候选", "进入 IHC / 空间验证\n例：SLPI（I=0.159，p=0.001）", "#E8F5E9", "#2E7D32", 52.0),
    ("组成信号", "归为组成驱动\n仅作分层变量使用", "#E1F5FE", "#0277BD", 36.5),
    ("管线缺陷", "先修管线，再谈生物学\n见右侧 F1–F6 检查表", "#FFEBEE", "#C62828", 21.0),
]
OX, OW, OH = 60.0, 32.0, 11.0
for name, desc, fc, ec, cy in outs:
    box(OX, cy - OH / 2, OW, OH, f"{name}\n{desc}", fc, ec, 9.6, lsp=1.5)
    arrow(AX_R + 0.3, AX_MID, OX - 0.5, cy)

# ---------------- 右：六类静默故障检查表 ----------------
FAX, FAW = 95.5, 36.0
box(FAX, 3.6, FAW, 57.9,
    "六类静默故障检查表\n（全部无报错 · 外观合理）\n"
    "\nF1  折级统计      → 只报 pooled OOF\n"
    "F2  NaN 污染协变量 → 断言非空率\n"
    "F3  CLI 标志被忽略 → 日志核验生效项\n"
    "F4  特征文件错配  → 特征 MD5 钉住\n"
    "F5  多线程漂移    → 单线程钉住重跑\n"
    "F6  分支空操作    → 跨配置输出 diff\n"
    "\nF1、F5 已量化\n其余四类：定性记录 + 检测启发式",
    "#F3E5F5", "#6A1B9A", 9.0, lsp=1.62)

# ---------------- 底部：关键数字 ----------------
bar_lines = (
    "应用演示：四个 CPTAC 队列（NSCLC 203 · PDA 120 · UCEC 46 · GBM 54）\n"
    "A2 协变量阶梯 −97%（100→3，亚型单独贡献 96%）   |   A3 字典来源 20× 摆动（100 vs 5）\n"
    "A1 折级统计多报 88%（40→5，57% 反向）   |   A5 片内置换 83 → 29 组成 + 54 排列候选\n"
    "F5 多线程 190 vs 181 命中（单线程后逐项复现）   |   A5 患者置换族内校准 p 0.05%（1/2,000）"
)
box(2.0, 3.6, 90.0, 10.6, bar_lines, "#ECEFF1", "#546E7A", 9.0, lsp=1.62)

ax.text(W / 2, 1.3, "判据预注册  ·  单线程钉住（OPENBLAS/OMP/MKL/NUMEXPR=1，seed=2026）  ·  双空对照缺一不可",
        ha="center", va="center", fontsize=9.6, color="#37474F", style="italic")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
TMP = OUT + ".tmp.png"
fig.savefig(TMP, bbox_inches="tight", facecolor="white")
plt.close(fig)

# 贴到固定画布（白底居中），保证每次输出与投稿版同尺寸
from PIL import Image
im = Image.open(TMP).convert("RGB")
canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
canvas.paste(im, ((CANVAS_W - im.width) // 2, (CANVAS_H - im.height) // 2))
canvas.save(OUT, dpi=(200, 200))
os.remove(TMP)
print("OK", OUT, os.path.getsize(OUT))
