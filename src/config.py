"""
config.py
=========
experiment2_fm_time — Foundation Model TIME 分解 + 分子标志物预测
全局配置：路径、模型、参数、队列映射。
"""
import os
from pathlib import Path

# ==============================================================
# 基础路径
# ==============================================================
ROOT = Path(__file__).parent.parent.resolve()
SRC_EXP1 = ROOT.parent / "experiment1"
SRC_CANVAS = SRC_EXP1 / "archive_plan1_canvas"

# 数据路径（部分复用 experiment1 已有数据）
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_FEAT = ROOT / "data" / "features"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

for d in [DATA_RAW, DATA_PROC, DATA_FEAT, RESULTS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# experiment1 已有数据的软链接/引用
# 自动探测：优先 experiment1 目录, 其次常见服务器位置
def _find_tcga_wsi():
    candidates = [
        ROOT / "data" / "raw" / "TCGA" / "svs",          # 首选: 你放的位置
        ROOT.parent / "experiment1" / "data" / "wsi",
        ROOT.parent / "experiment1" / "data" / "processed" / "tcga_wsi",
        ROOT / "data" / "raw" / "TCGA" / "wsi",
        ROOT / "data" / "TCGA_svs",
        Path("/root/autodl-tmp/data/TCGA_svs"),
        Path("/root/data/TCGA_svs"),
        Path("/home/user/data/TCGA_svs"),
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.svs")):
            print(f"[config] 找到 TCGA WSI: {c}  ({len(list(c.glob('*.svs')))} .svs)")
            return c
    # 都没找到就用默认（会在 patch_embed 里报清晰错误）
    print(f"[config] WARNING: 没找到 TCGA .svs, 预期位置: {candidates[0]}")
    return candidates[0]

SRC_TCGA_WSI = _find_tcga_wsi()
SRC_TCGA_PATCHES = SRC_EXP1 / "data" / "processed" / "tcga_wsi_patches"
SRC_TCGA_CLIN = SRC_EXP1 / "data" / "raw" / "TCGA" / "tcga_clinical.csv"
SRC_CPTAC_LUAD = SRC_EXP1 / "data" / "raw" / "CPTAC" / "CPTAC-LUAD_clinical.csv"
SRC_CPTAC_LUSC = SRC_EXP1 / "data" / "raw" / "CPTAC" / "CPTAC-LUSC_clinical.csv"
SRC_NICHE_OLD = SRC_EXP1 / "data" / "processed" / "niche_maps"  # MUSK 版(仅参考)

# CPTAC WSI 下载目标（新项目自己下）
CPTAC_WSI_DIR = DATA_RAW / "CPTAC" / "wsi"
TCGA_WSI_DIR = DATA_RAW / "TCGA" / "wsi"

# ==============================================================
# 模型配置（全部 DINOv2，无需 MUSK 授权）
# ==============================================================
BACKBONES = {
    "dinov2_vitb14": {
        "name": "dinov2_vitb14",      # ViT-B/14, 22M params, 384-dim
        "embed_dim": 384,
    },
    "dinov2_vitl14": {
        "name": "dinov2_vitl14",      # ViT-L/14, 303M params, 1024-dim
        "embed_dim": 1024,
    },
    "resnet50": {
        "name": "resnet50",           # ImageNet supervised, 25M params, 2048-dim
        "embed_dim": 2048,
    },
    "vit_b16": {
        "name": "vit_b16",            # ImageNet supervised ViT-B/16, 86M params, 768-dim
        "embed_dim": 768,
    },
}
DEFAULT_BACKBONE = "dinov2_vitl14"  # 主实验用 DINOv2-L

# ==============================================================
# WSI 处理参数
# ==============================================================
PATCH_SIZE = 224
MAGNIFICATION = 20
TISSUE_THRESHOLD = 0.5
MAX_PATCHES_PER_WSI = 2000

# ==============================================================
# TIME 分解参数
# ==============================================================
N_NICHES = 10                     # 10 类生态位（复用 archive 的定义）
NICHE_NAMES = [
    "tumor_core", "macrophage", "b_cell", "fibroblast_CAF",
    "neutrophil_inflam", "t_cell_immune", "tumor_immune_interface",
    "stroma", "necrosis", "vessel",
]
KMEANS_N_INIT = 20
SEED = 2026

# ==============================================================
# 下游任务
# ==============================================================
MOLECULAR_TASKS = [
    "EGFR_mutation", "KRAS_mutation", "TP53_mutation",
    "ALK_fusion", "RET_fusion", "ROS1_fusion",
    "TMB_high", "CIMP_status",
]
SURVIVAL_TASKS = ["OS_1yr", "OS_2yr", "OS_5yr", "Cox_regression"]

# ==============================================================
# 队列定义
# ==============================================================
COHORTS = {
    "TCGA": {
        "wsi_dir": TCGA_WSI_DIR,
        "clinical": DATA_RAW / "TCGA" / "tcga_clinical_full.csv",  # 补完突变的全量
        "split": "internal",       # 内部 CV
    },
    "CPTAC": {
        "wsi_dir": CPTAC_WSI_DIR,
        "clinical": DATA_RAW / "CPTAC" / "cptac_clinical_merged.csv",
        "split": "external",       # 外部验证
    },
}

# ==============================================================
# 输出目录结构
# ==============================================================
def embed_cache_dir(backbone: str) -> Path:
    return DATA_PROC / f"{backbone}_embeddings"

def niche_cache_dir(backbone: str) -> Path:
    return DATA_PROC / f"{backbone}_niches"

def features_dir(backbone: str) -> Path:
    return DATA_FEAT / f"{backbone}_features"

def kg_dir(backbone: str) -> Path:
    return RESULTS / backbone / "kg"

def insight_dir(backbone: str) -> Path:
    return RESULTS / backbone / "insight"
