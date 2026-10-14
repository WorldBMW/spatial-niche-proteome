"""
patch_embed.py
==============
从 WSI .svs 切片中提取每个 patch 的 foundation model embedding。

支持 backbone: dinov2_vitb14, dinov2_vitl14, resnet50, vit_b16
全部在 GPU 上跑，WSI 用 openslide 读取。

缓存格式：每个 WSI 一个 .npy, shape = (N_patches, embed_dim)
同时保存 coords.csv 记录每个 patch 的 (x, y) 坐标。
"""
import sys, os, argparse, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
import config
from case_id import resolve_case_id


# ==============================================================
# 本地权重目录（项目内 data/weights/）
# ==============================================================
LOCAL_WEIGHTS = Path(__file__).parent.parent / "data" / "weights"


def _local_timm_weight(timm_name: str) -> Path | None:
    """查本地有没有 timm 权重 (.safetensors)。"""
    p = LOCAL_WEIGHTS / f"timm-{timm_name}" / "model.safetensors"
    return p if p.exists() else None


def _local_torch_weight(filename: str) -> Path | None:
    """查本地有没有 torchvision 权重 (.pth)。"""
    p = LOCAL_WEIGHTS / "torch" / filename
    return p if p.exists() else None


# ==============================================================
# Backbone 加载
# ==============================================================
def load_backbone(name: str, device: str = "cuda"):
    """
    加载 backbone，优先从项目 data/weights/ 离线加载，
    找不到再 fallback 联网下载。返回 (model, transform, embed_dim)。
    """
    if name.startswith("dinov2"):
        import timm
        import safetensors.torch

        timm_map = {
            "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
            "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
        }
        timm_name = timm_map.get(name)
        if timm_name is None:
            raise ValueError(f"Unknown DINOv2: {name}. 支持 {list(timm_map.keys())}")

        local_weight = _local_timm_weight(timm_name)
        if local_weight:
            # --- 离线: 手动加载本地 .safetensors ---
            # 关键: 不传 img_size，让模型用预训练默认 (518×518) 的 pos_embed shape
            # forward 时 dynamic_img_size=True 会自动插值 pos_embed 到实际输入尺寸 (224×224)
            print(f"  [weight] 从本地加载: {local_weight}")
            model = timm.create_model(
                timm_name, pretrained=False, num_classes=0,
                dynamic_img_size=True,
            )
            state = safetensors.torch.load_file(str(local_weight))
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"    [WARN] missing keys: {missing[:3]}...")
            if unexpected:
                print(f"    [WARN] unexpected keys: {unexpected[:3]}...")
            # 手动补 pretrained_cfg (timm.data 的默认 ImageNet mean/std)
            from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            model.pretrained_cfg = {"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD}
        else:
            # --- fallback: 联网 ---
            print(f"  [weight] 本地未找到，尝试联网下载...")
            model = timm.create_model(
                timm_name, pretrained=True, num_classes=0,
                dynamic_img_size=True,
            )

        embed_dim = model.embed_dim
        cfg = getattr(model, "pretrained_cfg", {})
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        print(f"  [DINOv2 via timm] {timm_name} embed_dim={embed_dim}")

    elif name == "resnet50":
        import torchvision.models as tvm
        local_w = _local_torch_weight("resnet50-11ad3fa6.pth")
        if local_w:
            print(f"  [weight] 从本地加载: {local_w}")
            model = tvm.resnet50(weights=None)
            state = torch.load(str(local_w), map_location="cpu")
            model.load_state_dict(state)
        else:
            print(f"  [weight] 本地未找到，尝试联网下载...")
            model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
        embed_dim = 2048
        transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    elif name == "vit_b16":
        import torchvision.models as tvm
        local_w = _local_torch_weight("vit_b_16-c867db91.pth")
        if local_w:
            print(f"  [weight] 从本地加载: {local_w}")
            model = tvm.vit_b_16(weights=None)
            state = torch.load(str(local_w), map_location="cpu")
            model.load_state_dict(state)
        else:
            print(f"  [weight] 本地未找到，尝试联网下载...")
            model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()
        embed_dim = 768
        transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    else:
        raise ValueError(f"Unknown backbone: {name}")

    model.eval().to(device)
    print(f"  [patch_embed] dev={device}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, transform, embed_dim


# ==============================================================
# WSI patch 提取 (优化版: block-based, 减少 OpenSlide 调用)
# ==============================================================
def extract_patches_from_svs(svs_path: Path, patch_size: int = 224,
                              magnification: int = 20,
                              tissue_threshold: float = 0.05,
                              max_patches: int = 2000):
    """
    从 .svs 提取 patch (PIL.Image) + 坐标。
    优化: 按 block 读 OpenSlide (每块 2048×2048), 块内切块 → 大幅减少 JPEG2000 decode 次数。
    旧版逐 tile 读: O(12K) 次 OpenSlide 调用 → 22min/WSI
    新版 block 读:   O(144) 次 OpenSlide 调用 → ~20s/WSI
    """
    try:
        import openslide
    except ImportError:
        raise ImportError("需先 pip install openslide-python")

    slide = openslide.OpenSlide(str(svs_path))
    objective = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
    target_downsample = objective / magnification
    level = slide.get_best_level_for_downsample(target_downsample)
    level_downsample = slide.level_downsamples[level]

    W, H = slide.level_dimensions[level]
    step = int(patch_size * target_downsample / level_downsample)
    w_n = W // step
    h_n = H // step
    total_tiles = w_n * h_n

    # block_size: 读大块再切小块
    BLOCK = 2048
    patches, coords = [], []

    for by in range(0, H, BLOCK):
        for bx in range(0, W, BLOCK):
            bw = min(BLOCK, W - bx)
            bh = min(BLOCK, H - by)
            try:
                block = np.array(slide.read_region(
                    (int(bx * level_downsample), int(by * level_downsample)),
                    level, (bw, bh)))
            except Exception:
                continue  # 个别 block 损坏跳过

            # 块内切块 (跳过边缘不足 step 的)
            for iy_off in range(0, bh - step + 1, step):
                for ix_off in range(0, bw - step + 1, step):
                    tile = block[iy_off:iy_off + step, ix_off:ix_off + step, :3]
                    if tile.shape[:2] != (step, step):
                        continue
                    # tissue 过滤
                    mean_v = tile[:, :, :3].mean() / 255.0
                    if 0.05 < mean_v < 0.95:
                        # resize 到 patch_size (如果 step != patch_size)
                        if step != patch_size:
                            from PIL import Image
                            patch = Image.fromarray(tile).resize((patch_size, patch_size), Image.BILINEAR)
                        else:
                            from PIL import Image
                            patch = Image.fromarray(tile)
                        # 算 level0 坐标
                        x0 = int((bx + ix_off) * level_downsample)
                        y0 = int((by + iy_off) * level_downsample)
                        patches.append(patch.convert("RGB"))
                        coords.append((x0, y0))
                        if len(patches) >= max_patches:
                            slide.close()
                            return patches, coords
            if len(patches) >= max_patches:
                slide.close()
                return patches, coords

    slide.close()
    return patches, coords


# ==============================================================
# 批量 embedding
# ==============================================================
@torch.no_grad()
def embed_patches(patches, model, transform, device, batch_size=128, use_amp=True):
    """把 PIL.Image 列表转成 embedding matrix (N, dim)。"""
    feats = []
    # 半精度 AMP —— GPU 速度翻倍，显存省一半
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and device == "cuda") else torch.no_grad()
    with amp_ctx:
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            tens = torch.stack([transform(p) for p in batch]).to(device, non_blocking=True)
            f = model(tens)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, model.embed_dim))


# ==============================================================
# 主入口
# ==============================================================
def process_wsi(svs_path: Path, out_dir: Path, model, transform, embed_dim,
                device: str = "cuda") -> bool:
    """处理单个 WSI，存 embedding + coords。model/transform 外部传入（避免每 WSI 重加载）。"""
    case_id = resolve_case_id(svs_path.stem)
    out_embed = out_dir / f"{case_id}.npy"
    out_coords = out_dir / f"{case_id}_coords.csv"
    if out_embed.exists() and out_coords.exists():
        return True  # skip

    try:
        patches, coords = extract_patches_from_svs(
            svs_path, config.PATCH_SIZE, config.MAGNIFICATION,
            config.TISSUE_THRESHOLD, config.MAX_PATCHES_PER_WSI,
        )
        if len(patches) < 10:
            print(f"  [SKIP] {case_id}: only {len(patches)} patches")
            return False

        emb = embed_patches(patches, model, transform, device)
        np.save(out_embed, emb)
        pd.DataFrame(coords, columns=["x", "y"]).to_csv(out_coords, index=False)
        return True
    except Exception as e:
        print(f"  [ERROR] {case_id}: {e}")
        return False
    finally:
        gc.collect()
        # 不要每 WSI empty_cache —— 反而拖慢


def run(backbone: str = None, wsi_dir: Path = None, out_dir: Path = None):
    backbone = backbone or config.DEFAULT_BACKBONE
    wsi_dir = Path(wsi_dir) if wsi_dir else config.SRC_TCGA_WSI
    out_dir = Path(out_dir) if out_dir else config.embed_cache_dir(backbone)
    out_dir.mkdir(parents=True, exist_ok=True)

    svs_files = sorted(wsi_dir.glob("*.svs"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[patch_embed] backbone={backbone}  WSI={len(svs_files)}  out={out_dir}  dev={device}")

    # === 关键优化: 只加载一次模型 ===
    t0 = time.time()
    model, transform, embed_dim = load_backbone(backbone, device)
    if device == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"  [torch.compile] OK")
        except Exception as e:
            print(f"  [torch.compile] skip: {e}")
    print(f"  [模型加载] {time.time()-t0:.1f}s")

    # === 动态 batch_size —— 根据 GPU 显存自动调 ===
    if device == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        if total_mem >= 40:     # A100 40GB/80GB
            batch_size = 256
        elif total_mem >= 16:   # 3090/4090 24GB
            batch_size = 192
        else:                   # 12GB 以下
            batch_size = 128
        print(f"  [batch_size] {batch_size} (GPU {total_mem:.1f}GB)")
    else:
        batch_size = 64

    ok = 0
    t_start = time.time()
    for svs in tqdm(svs_files, desc=f"embedding[{backbone}]"):
        if process_wsi(svs, out_dir, model, transform, embed_dim, device):
            ok += 1
    dt = time.time() - t_start
    print(f"[patch_embed] done. {ok}/{len(svs_files)} succeeded.  "
          f"total={dt/60:.1f}min  avg={dt/max(ok,1):.1f}s/WSI")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.DEFAULT_BACKBONE,
                    choices=list(config.BACKBONES.keys()))
    ap.add_argument("--wsi-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.backbone, args.wsi_dir, args.out)
