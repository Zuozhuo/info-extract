import os
import math

ROOT_DIR = "/home/zuozhuo/info-extract/output"
MAX_CHARS = 8000

def split_md(md_path, max_chars=MAX_CHARS):
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text:
        print(f"[SKIP] 空文件: {md_path}")
        return

    n_chunks = math.ceil(len(text) / max_chars)
    base_dir = os.path.dirname(md_path)
    pdf_name = os.path.splitext(os.path.basename(md_path))[0]

    for i in range(n_chunks):
        chunk_text = text[i * max_chars : (i + 1) * max_chars]
        chunk_path = os.path.join(base_dir, f"{pdf_name}_chunk{i+1}.md")

        # 如果已经存在，就跳过，避免重复切
        if os.path.exists(chunk_path):
            print(f"[EXIST] 跳过已存在: {chunk_path}")
            continue

        with open(chunk_path, "w", encoding="utf-8") as cf:
            cf.write(chunk_text)
        print(f"[WRITE] {chunk_path}  ({len(chunk_text)} chars)")

def main():
    for entry in os.listdir(ROOT_DIR):
        subdir = os.path.join(ROOT_DIR, entry)
        if not os.path.isdir(subdir):
            continue

        # 子文件夹名 = pdf 名称
        pdf_name = entry
        md_path = os.path.join(subdir, "auto", f"{pdf_name}.md")

        if not os.path.exists(md_path):
            print(f"[MISS] 找不到 md: {md_path}")
            continue

        print(f"\n[PROC] {md_path}")
        split_md(md_path)

if __name__ == "__main__":
    main()
