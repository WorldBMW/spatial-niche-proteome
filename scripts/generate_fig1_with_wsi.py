"""
generate_fig1_with_wsi.py - 生成包含 WSI 图像的增强版 Fig 1
展示：WSI 切片 → patch 提取 → niche 聚类 → 五轴协议工作流
"""
import os
import glob
import numpy as np
from PIL import Image
import openslide
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# 选择一张 WSI（选 LUAD 相关的）
wsi_dir = "data/wsi"
wsi_files = sorted(glob.glob(os.path.join(wsi_dir, "*.svs")))
print(f"找到 {len(wsi_files)} 张 WSI")

# 选一张中等大小的 WSI
wsi_path = None
for f in wsi_files[:50]:
    try:
        s = openslide.OpenSlide(f)
        w, h = s.dimensions
        if 50000 < w < 120000 and 50000 < h < 120000:
            wsi_path = f
            s.close()
            break
        s.close()
    except:
        continue

if wsi_path is None:
    wsi_path = wsi_files[0]

print(f"使用 WSI: {os.path.basename(wsi_path)}")

# 提取缩略图
slide = openslide.OpenSlide(wsi_path)
thumb = slide.get_thumbnail((800, 600))
thumb_arr = np.array(thumb)
slide.close()
print(f"缩略图尺寸: {thumb_arr.shape}")

# 生成 Fig 1：WSI + 协议工作流
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3,
              height_ratios=[1, 0.8])

# A: WSI 缩略图 + patch 示意
ax_wsi = fig.add_subplot(gs[0, :2])
ax_wsi.imshow(thumb_arr)
ax_wsi.set_title("A. WSI (H&E, DINOv2 ViT-L/14)", fontsize=11, fontweight='bold')
ax_wsi.axis('off')
# 画几个 patch 示意框
h, w = thumb_arr.shape[:2]
for i, (x, y) in enumerate([(w//4, h//3), (w//2, h//2), (3*w//4, h//3)]):
    rect = patches.Rectangle((x-30, y-30), 60, 60, linewidth=2,
                             edgecolor=['#e74c3c', '#2ecc71', '#3498db'][i],
                             facecolor='none')
    ax_wsi.add_patch(rect)
    ax_wsi.text(x, y-35, f'patch {i+1}', ha='center', fontsize=8,
                color=['#e74c3c', '#2ecc71', '#3498db'][i], fontweight='bold')

# B: 五轴协议流程（简洁版）
ax_proto = fig.add_subplot(gs[0, 2:])
ax_proto.axis('off')
ax_proto.set_title("B. Five-axis stress-test protocol", fontsize=11, fontweight='bold')

# 画流程框图
boxes = [
    ("WSI + bulk omics", "#3498db", 0.05, 0.7),
    ("Association pipeline\n(ElasticNet, pooled OOF)", "#f39c12", 0.05, 0.45),
    ("Null controls\n(patient + spatial perm.)", "#e74c3c", 0.05, 0.2),
    ("Decision\ncomposition / arrangement / artifact", "#2ecc71", 0.55, 0.45),
]
for text, color, x, y in boxes:
    rect = patches.FancyBboxPatch((x, y), 0.38, 0.18, boxstyle="round,pad=0.02",
                                   facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
    ax_proto.add_patch(rect)
    ax_proto.text(x+0.19, y+0.09, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 箭头
ax_proto.annotate('', xy=(0.55, 0.54), xytext=(0.43, 0.54),
                  arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

# 五轴标签
axes_labels = ["A1: Statistical", "A2: Covariate", "A3: Dictionary", "A4: Feature type", "A5: Null controls"]
for i, label in enumerate(axes_labels):
    x_pos = 0.05 + i * 0.19
    ax_proto.text(x_pos, 0.03, label, ha='center', va='center', fontsize=7.5,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='#7f8c8d'))

# C: Niche 聚类示意（用模拟的 patch 嵌入聚类）
ax_niche = fig.add_subplot(gs[1, 0])
np.random.seed(42)
niche_colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6',
                '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
for k in range(10):
    x = np.random.normal(k % 5 * 2, 0.5, 50)
    y = np.random.normal(k // 5 * 2, 0.5, 50)
    ax_niche.scatter(x, y, c=niche_colors[k], s=10, alpha=0.6, label=f'niche {k+1}')
ax_niche.set_title("C. k-means niche (k=10)", fontsize=10, fontweight='bold')
ax_niche.set_xlabel("patch embedding dim 1", fontsize=8)
ax_niche.set_ylabel("patch embedding dim 2", fontsize=8)
ax_niche.tick_params(labelsize=7)

# D: 230 维空间生态特征示意
ax_feat = fig.add_subplot(gs[1, 1])
# 模拟特征矩阵热力图
feat_matrix = np.random.rand(10, 23)
im = ax_feat.imshow(feat_matrix, cmap='viridis', aspect='auto')
ax_feat.set_title("D. 230-d spatial features", fontsize=10, fontweight='bold')
ax_feat.set_xlabel("feature index", fontsize=8)
ax_feat.set_ylabel("niche", fontsize=8)
ax_feat.tick_params(labelsize=7)

# E: 组成 vs 排列分解示意
ax_decomp = fig.add_subplot(gs[1, 2])
categories = ['Composition\ndriven', 'Arrangement\ncandidate']
counts = [29, 54]
colors = ['#3498db', '#e74c3c']
bars = ax_decomp.bar(categories, counts, color=colors, alpha=0.7, edgecolor='#333')
for bar, count in zip(bars, counts):
    ax_decomp.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_decomp.set_title("E. Spatial permutation\ndecomposition (NSCLC)", fontsize=10, fontweight='bold')
ax_decomp.set_ylabel("# proteins", fontsize=8)
ax_decomp.tick_params(labelsize=8)

# F: 协议输出分诊
ax_triage = fig.add_subplot(gs[1, 3])
triage = ['Arrangement\ncandidate', 'Composition\ndriven', 'Pipeline\nartifact']
counts_t = [2, 29, 0]  # 亚型校正后
colors_t = ['#e74c3c', '#3498db', '#95a5a6']
bars_t = ax_triage.bar(triage, counts_t, color=colors_t, alpha=0.7, edgecolor='#333')
for bar, count in zip(bars_t, counts_t):
    label = str(count) if count > 0 else '0'
    ax_triage.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                   label, ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_triage.set_title("F. After full covariate\nadjustment", fontsize=10, fontweight='bold')
ax_triage.set_ylabel("# proteins", fontsize=8)
ax_triage.tick_params(labelsize=7)

plt.suptitle("Fig 1. Five-axis stress-test protocol for pathology-FM to bulk-omics associations",
             fontsize=13, fontweight='bold', y=0.98)

out_path = "papers/BIB_Original/figures/Fig1_protocol_schema_v2.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"已保存: {out_path}")
