"""
md_to_docx.py - 将 manuscript_CN.md 转换为 .docx
支持：标题、段落、表格、列表、加粗、斜体、代码引用
"""
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

MD_PATH = Path("papers/BIB_Original/manuscript_CN.md")
DOCX_PATH = Path("papers/BIB_Original/manuscript_CN_updated.docx")


def parse_inline(text):
    """解析行内格式：**bold**, *italic*, `code`"""
    parts = []
    # 按 **, *, ` 分割
    pattern = re.compile(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)')
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            parts.append(("plain", text[pos:m.start()]))
        token = m.group()
        if token.startswith("**"):
            parts.append(("bold", token[2:-2]))
        elif token.startswith("`"):
            parts.append(("code", token[1:-1]))
        else:
            parts.append(("italic", token[1:-1]))
        pos = m.end()
    if pos < len(text):
        parts.append(("plain", text[pos:]))
    return parts


def add_inline_runs(paragraph, text):
    for kind, content in parse_inline(text):
        run = paragraph.add_run(content)
        if kind == "bold":
            run.bold = True
        elif kind == "italic":
            run.italic = True
        elif kind == "code":
            run.font.name = "Consolas"
            run.font.size = Pt(10)


def parse_md_table(lines, start_idx):
    """解析 Markdown 表格，返回 (headers, rows, end_idx)"""
    headers = [c.strip() for c in lines[start_idx].strip().strip("|").split("|")]
    # 跳过分隔行
    rows = []
    i = start_idx + 2
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(row)
        i += 1
    return headers, rows, i


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = Document()

    # 设置默认字体
    style = doc.styles["Normal"]
    font = style.font
    font.name = "宋体"
    font.size = Pt(11)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        # 空行
        if not stripped:
            i += 1
            continue

        # 表格
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r'^\|[\s\-:|]+\|$', lines[i+1].strip()):
            headers, rows, end_idx = parse_md_table(lines, i)
            table = doc.add_table(rows=1 + len(rows), cols=len(headers))
            table.style = "Table Grid"
            # 表头
            for j, h in enumerate(headers):
                cell = table.rows[0].cells[j]
                cell.text = ""
                p = cell.paragraphs[0]
                run = p.add_run(h)
                run.bold = True
            # 数据行
            for r_idx, row in enumerate(rows):
                for j, val in enumerate(row):
                    cell = table.rows[r_idx + 1].cells[j]
                    cell.text = ""
                    p = cell.paragraphs[0]
                    add_inline_runs(p, val)
            doc.add_paragraph()
            i = end_idx
            continue

        # 标题
        if stripped.startswith("# "):
            p = doc.add_heading(stripped[2:], level=0)
            i += 1
            continue
        if stripped.startswith("## "):
            p = doc.add_heading(stripped[3:], level=1)
            i += 1
            continue
        if stripped.startswith("### "):
            p = doc.add_heading(stripped[4:], level=2)
            i += 1
            continue
        if stripped.startswith("#### "):
            p = doc.add_heading(stripped[5:], level=3)
            i += 1
            continue

        # 无序列表
        if re.match(r'^[-*]\s', stripped):
            p = doc.add_paragraph(style="List Bullet")
            add_inline_runs(p, re.sub(r'^[-*]\s+', '', stripped))
            i += 1
            continue

        # 有序列表（参考文献章节保留原始编号文本，不依赖Word列表样式）
        if re.match(r'^\d+\.\s', stripped):
            p = doc.add_paragraph()
            add_inline_runs(p, stripped)
            i += 1
            continue

        # 普通段落
        p = doc.add_paragraph()
        add_inline_runs(p, stripped)
        i += 1

    # 在参考文献之前插入图片（如果存在 figures 目录）
    fig_dir = Path("papers/BIB_Original/figures")
    if fig_dir.exists():
        # 找到参考文献段落，在其之前插入图片
        ref_para = None
        for para in doc.paragraphs:
            if "参考文献" in para.text.strip():
                ref_para = para
                break

        # 图注映射（与稿件图注对应）
        fig_files = {
            1: "Fig1_protocol_schema_v2.png",
            2: "Fig2_nslc_attenuation.png",
            3: "Fig3_cross_cancer.png",
            4: "Fig4_protocol_boundary.png",
            5: "Fig5_SLPI_spatial_validation.png",
        }

        if ref_para is not None:
            # 找到参考文献段落的索引
            insert_idx = None
            for idx, para in enumerate(doc.paragraphs):
                if para is ref_para:
                    insert_idx = idx
                    break
            if insert_idx is None:
                insert_idx = len(doc.paragraphs) - 1
            fig_captions = {
                1: "图 1. 五轴压力测试协议工作流。(A) 代表性 H&E 整张切片（WSI）及 patch 提取示意；(B) 五轴协议流程：WSI+bulk 组学 → 关联管线（ElasticNet, pooled OOF）→ 双空对照 → 组成/排列/伪影分诊；(C) k-means niche 聚类（k=10）；(D) 230 维空间生态特征矩阵；(E) 空间置换分解：29 组成驱动型 + 54 排列候选型；(F) 全协变量校正后仅 2 个排列特异蛋白存活。",
                2: "图 2. NSCLC 与 PDA 的跨队列衰减观察。(A) NSCLC 协变量阶梯：无校正 100 命中 → age+sex 83 → +subtype 3（总衰减 97%）；(B) 字典来源摆动：冻结式（TCGA）100 vs 转导式 5（20 倍）；(C) 特征类型对比：空间 83 vs 池化 190，亚型校正后 3 vs 9；(D) 组学平台：蛋白 83 vs RNA 55 vs 磷酸化 0。",
                3: "图 3. PDA 归因与跨队列结果。(A) PDA 三 run 归因设计：no-cov(n=120) 17 → no-cov(n=65) 13 → composition-cov(n=65) 6，组成协变量吸收 54%；(B) PDA 新命中在组成校正下的命运；(C) PDA 空间置换分解：6 组成驱动 + 0 排列特异；(D) PDA 特征类型消融。",
                4: "图 4. 静默故障检测与协议边界。(A) 患者置换空模型：经验 FDR=0.05%（1/2000），仅 ANK3 跨越阈值；(B) 统计协议轴：折级 t 检验 40 命中 vs pooled 精确检验 5 命中（−88%）。",
                5: "图 5. SLPI 独立空间验证（LUAD Visium 切片，GSE189487，4,095 spots）。左：SLPI log1p(CPM) 空间分布，Moran's I=0.159，p_perm=0.001；中：SLPI raw counts；右：SLPI 表达分布直方图。注：单切片、单基因，属必要非充分外部证据。",
            }

            from docx.shared import Inches
            insert_pos = insert_idx
            for fig_num in sorted(fig_files.keys()):
                fig_path = fig_dir / fig_files[fig_num]
                if fig_path.exists():
                    # 插入图片标题
                    p_title = doc.paragraphs[insert_pos].insert_paragraph_before(f"Figure {fig_num}")
                    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in p_title.runs:
                        run.bold = True
                    insert_pos += 1
                    # 插入图片
                    p_img = doc.paragraphs[insert_pos].insert_paragraph_before()
                    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    run = p_img.add_run()
                    run.add_picture(str(fig_path), width=Inches(6.0))
                    insert_pos += 1
                    # 插入图注
                    p_cap = doc.paragraphs[insert_pos].insert_paragraph_before(fig_captions[fig_num])
                    p_cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    insert_pos += 1

    doc.save(str(DOCX_PATH))
    print(f"saved: {DOCX_PATH}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        MD_PATH = Path(sys.argv[1])
        DOCX_PATH = Path(sys.argv[2])
    main()
