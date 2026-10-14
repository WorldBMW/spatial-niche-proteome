"""
case_id.py
==========
统一 case_id 解析：从 .svs 文件名提取真实 TCGA barcode 或 CPTAC ID。

WSI 文件名格式示例：
  TCGA-05-4403-01-01-TS1.4ea11b71-999c-4073-ab56-c6da0eb310f8.svs
  → TCGA barcode = "TCGA-05-4403" (前 3 段)

  C3L-00001.svs (LUAD), C3N-00247.svs (LSCC) 或 CPTAC-S00001.svs
  → CPTAC ID = "C3L-00001" / "C3N-00247" / "CPTAC-S00001"

设计原则：
  - 纯函数，无副作用，方便单元测试
  - 识别多种命名习惯（有的 .svs 带 UUID 后缀，有的不带）
  - 保持返回值稳定：同一个输入永远输出同一个 case_id
"""
import math
import re
from pathlib import Path


def pd_is_nan(x) -> bool:
    """安全判断 pandas NaN (float NaN)。"""
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False

# ==============================================================
# 正则模式（按优先级尝试）
# ==============================================================

# TCGA barcode: TCGA-XX-XXXX (严格 4 位 + 4 位数字)
_RE_TCGA = re.compile(r"(TCGA-\d{2}-\d{4})", re.IGNORECASE)

# CPTAC 患者 ID: C3L-XXXXX (LUAD) 或 C3N-XXXXX (LSCC)
_RE_CPTAC_C3X = re.compile(r"(C3[LN]-\d{5})", re.IGNORECASE)

# CPTAC ID 变体 2: CPTAC-SXXXXX 或 CPTAC-XXXXX
_RE_CPTAC_PREFIX = re.compile(r"(CPTAC[-_][A-Z]?\d{4,6})", re.IGNORECASE)

# MSK-IMPACT / MSK-NSCLC 变体
_RE_MSK = re.compile(r"(MSK[-_A-Z0-9]+)", re.IGNORECASE)

# UUID (36 chars, 标准格式)
_RE_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def parse_tcga_barcode(filename: str) -> str | None:
    """从文件名提取 TCGA barcode (TCGA-XX-XXXX)。找不到返回 None。"""
    # 去掉扩展名
    stem = Path(filename).stem
    m = _RE_TCGA.search(stem)
    if m:
        # 统一大写
        return m.group(1).upper()
    return None


def parse_cptac_id(filename: str) -> str | None:
    """从文件名提取 CPTAC ID。找不到返回 None。"""
    stem = Path(filename).stem
    m = _RE_CPTAC_C3X.search(stem)
    if m:
        return m.group(1).upper()
    m = _RE_CPTAC_PREFIX.search(stem)
    if m:
        return m.group(1).upper()
    return None


def resolve_case_id(filename) -> str:
    """
    自动识别文件名来源并返回统一 case_id。
    
    优先级: TCGA barcode → CPTAC ID → MSK → 原始 stem (fallback)
    
    Args:
        filename: .svs 文件名或路径 (如 "TCGA-05-4403-01-01-TS1.xxx.svs"
                  或完整路径 "/data/wsi/C3L-00001.svs")
                  也可以是 float NaN / None (pandas apply 里可能出现)
    
    Returns:
        统一格式的 case_id；若输入无效返回空字符串
    
    Examples:
        >>> resolve_case_id("TCGA-05-4403-01-01-TS1.4ea11b71.svs")
        'TCGA-05-4403'
        >>> resolve_case_id("C3L-00001.svs")
        'C3L-00001'
        >>> resolve_case_id(float('nan'))
        ''
    """
    # 防御 NaN / None / float —— pandas apply 里 case_id 列可能有空值
    if filename is None:
        return ""
    if isinstance(filename, float) and pd_is_nan(filename):
        return ""
    # 确保是字符串
    filename = str(filename).strip()
    if not filename or filename.lower() == "nan":
        return ""
    # 先提取 stem (去掉路径和扩展名)
    stem = Path(filename).stem
    
    # 按优先级尝试
    tcga = parse_tcga_barcode(stem)
    if tcga:
        return tcga
    
    cptac = parse_cptac_id(stem)
    if cptac:
        return cptac
    
    # fallback: 用原始 stem 的前 8 个字符（去掉 UUID 后缀）
    # 但去掉 .svs 可能带的 UUID 后缀
    clean = stem
    # 如果 stem 里有 TCGA- 但没匹配到完整 barcode (不太可能), 就返回前 12 字符
    if "TCGA" in stem.upper():
        # 尝试截取到第四个 "-" 之前
        parts = stem.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3]).upper()
    
    return clean


def is_uuid(s: str) -> bool:
    """判断字符串是否是 UUID 格式 (36 字符)。"""
    return bool(_RE_UUID.match(s))


# ==============================================================
# 批量处理工具
# ==============================================================

def remap_case_ids_inplace(df, case_id_col: str = "case_id"):
    """
    对 DataFrame 的 case_id 列做就地解析替换。
    
    Args:
        df: 含 case_id 列的 DataFrame
        case_id_col: case_id 列名
    
    Returns:
        新 DataFrame (不修改原 df)，case_id 列已替换
    
    打印映射摘要：解析了多少个，多少个是 fallback。
    """
    df = df.copy()
    original = df[case_id_col].tolist()
    resolved = [resolve_case_id(str(x)) for x in original]
    
    n_fallback = sum(1 for o, r in zip(original, resolved) if o != r and not (
        _RE_TCGA.search(str(o)) or _RE_CPTAC_C3X.search(str(o)) or _RE_CPTAC_PREFIX.search(str(o))
    ))
    
    df[case_id_col] = resolved
    print(f"[case_id] remapped {len(df)} entries: "
          f"{len(df) - n_fallback} parsed OK, {n_fallback} fallback")
    if n_fallback > 0:
        sample_orig = [o for o, r in zip(original, resolved) if o != r and not (
            _RE_TCGA.search(str(o)) or _RE_CPTAC_C3X.search(str(o)) or _RE_CPTAC_PREFIX.search(str(o))
        )][:3]
        print(f"  fallback examples: {sample_orig}")
    return df


# ==============================================================
# 快速测试
# ==============================================================
if __name__ == "__main__":
    tests = [
        ("TCGA-05-4403-01-01-TS1.4ea11b71-999c-4073-ab56-c6da0eb310f8", "TCGA-05-4403"),
        ("TCGA-18-3408-01-01-TS1.eb6d8014-eaf9-4679-955f-8ad4c5205850", "TCGA-18-3408"),
        ("TCGA-05-4244-01-01-TS1", "TCGA-05-4244"),
        ("C3L-00001", "C3L-00001"),
        ("C3L-02745", "C3L-02745"),
        ("C3N-00247__C3N-00247-21", "C3N-00247"),
        ("C3N-01030__C3N-01030-23.svs", "C3N-01030"),
        ("CPTAC-S00001", "CPTAC-S00001"),
        ("/path/to/TCGA-05-4403-01-01-TS1.svs", "TCGA-05-4403"),
    ]
    
    print("=== case_id 单元测试 ===")
    passed = 0
    for inp, expected in tests:
        result = resolve_case_id(inp)
        ok = result == expected
        passed += ok
        status = "✓" if ok else "✗"
        print(f"  {status} {inp[:50]:<50} → {result:<15} (expected {expected})")
    
    print(f"\n结果: {passed}/{len(tests)} passed")
    if passed < len(tests):
        raise SystemExit(1)
