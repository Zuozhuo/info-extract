# 批量把合订本 PDF 按目录书签切成“一篇一 PDF”
# 导出命名规则: <原始pdf名>_001.pdf, <原始pdf名>_002.pdf ...
# 依赖: pip install pymupdf 或 pip install PyPDF2

import os
from pathlib import Path

BASE_DIR = Path(r"/home/zuozhuo/info-extract/zr_pdfs")  # ← 修改为你的目录
RECURSIVE = True   # 是否递归子目录

# ---------- 方案 A：PyMuPDF ----------
def split_with_pymupdf(pdf_path: Path):
    import fitz
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)
    if not toc:
        return False, "无书签"

    # 默认优先 L1，如果不够则尝试 L2
    def pick_entries(level):
        return [(lvl, title.strip(), p) for (lvl, title, p) in toc if lvl == level]
    entries = pick_entries(1)
    if len(entries) < 2:
        alt = pick_entries(2)
        if len(alt) >= 2:
            entries = alt

    if not entries:
        return False, "未找到合适层级的书签"

    out_dir = pdf_path.with_suffix("")
    out_dir.mkdir(exist_ok=True)

    starts = [p for (_, _, p) in entries]
    ends = starts[1:] + [doc.page_count]

    base_name = pdf_path.stem
    count = 0
    for idx, (start, end) in enumerate(zip(starts, ends), 1):
        start0, end0 = max(0, start-1), max(0, end-1)
        sub = fitz.open()
        sub.insert_pdf(doc, from_page=start0, to_page=end0)
        out_file = out_dir / f"{base_name}_{idx:03d}.pdf"
        sub.save(out_file)
        sub.close()
        count += 1
        print(f"✅ [{pdf_path.name}] -> {out_file.name}  (p.{start}–{end})")

    doc.close()
    return True, f"{count} 篇"

# ---------- 方案 B：PyPDF2 ----------
def split_with_pypdf2(pdf_path: Path):
    from PyPDF2 import PdfReader, PdfWriter
    reader = PdfReader(str(pdf_path))

    outlines = None
    if hasattr(reader, "outline"):
        outlines = reader.outline
    elif hasattr(reader, "outlines"):
        outlines = reader.outlines
    else:
        try:
            outlines = reader.getOutlines()
        except Exception:
            outlines = None
    if not outlines:
        return False, "无书签"

    if hasattr(reader, "get_destination_page_number"):
        get_page = reader.get_destination_page_number
    elif hasattr(reader, "getDestinationPageNumber"):
        get_page = reader.getDestinationPageNumber
    else:
        return False, "PyPDF2 版本不支持书签页码"

    rows = []
    def walk(items, level=1):
        for entry in items:
            if isinstance(entry, list):
                walk(entry, level+1)
            else:
                title = getattr(entry, "title", str(entry)).strip()
                try:
                    p1 = int(get_page(entry)) + 1
                except Exception:
                    continue
                rows.append((level, title, p1))
    walk(outlines, 1)

    entries = [(lvl, title, p) for (lvl, title, p) in rows if lvl == 1]
    if len(entries) < 2:
        alt = [(lvl, title, p) for (lvl, title, p) in rows if lvl == 2]
        if len(alt) >= 2:
            entries = alt

    if not entries:
        return False, "未找到合适层级的书签"

    out_dir = pdf_path.with_suffix("")
    out_dir.mkdir(exist_ok=True)

    starts = [p for (_, _, p) in entries]
    ends = starts[1:] + [len(reader.pages)]

    base_name = pdf_path.stem
    count = 0
    for idx, (start, end) in enumerate(zip(starts, ends), 1):
        writer = PdfWriter()
        for p in range(start-1, end):
            writer.add_page(reader.pages[p])
        out_file = out_dir / f"{base_name}_{idx:03d}.pdf"
        with open(out_file, "wb") as f:
            writer.write(f)
        count += 1
        print(f"✅ [{pdf_path.name}] -> {out_file.name}  (p.{start}–{end})")

    return True, f"{count} 篇"

# ---------- 批处理 ----------
pdf_list = list(BASE_DIR.rglob("*.pdf") if RECURSIVE else BASE_DIR.glob("*.pdf"))
if not pdf_list:
    print(f"在 {BASE_DIR} 未找到 PDF 文件")
else:
    print(f"在 {BASE_DIR} 发现 {len(pdf_list)} 个 PDF，开始处理…\n")

    total_ok = 0
    for i, pdf in enumerate(sorted(pdf_list), 1):
        print(f"[{i}/{len(pdf_list)}] 处理：{pdf}")
        ok = False
        try:
            ok, msg = split_with_pymupdf(pdf)
        except ImportError:
            print("未安装 PyMuPDF，回退 PyPDF2 …")
        except Exception as e:
            print(f"[PyMuPDF] 失败：{e}，回退 PyPDF2 …")

        if not ok:
            try:
                ok, msg = split_with_pypdf2(pdf)
            except ImportError:
                print("未安装 PyPDF2，请 pip install PyPDF2")
                msg = "缺少依赖"
            except Exception as e:
                msg = f"异常：{e}"

        if ok:
            total_ok += 1
            print(f"🎉 完成：{pdf.name} -> {msg}\n")
        else:
            print(f"❌ 跳过：{pdf.name}（原因：{msg}）\n")

    print(f"✅ 全部完成：共 {len(pdf_list)} 个文件，成功 {total_ok} 个。")
