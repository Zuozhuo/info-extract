#!/usr/bin/env python3
import os
import re
import json
import pandas as pd

# 根路径配置
CSV_PATH = "/home/zuozhuo/info-extract/zr_output/final.csv"
OUTPUT_CSV_PATH = "/home/zuozhuo/info-extract/zr_output/final_with_images.csv"
MD_ROOT = "/home/zuozhuo/info-extract/output"

# 匹配 `![](images/xxx.jpg)` 和下一行 caption
# group1: 相对路径 images/xxx.jpg
# group2: 下一行 caption
IMG_PATTERN = re.compile(
    r'!\[\]\((images/[^)]+)\)\s*\r?\n\s*([^\n\r]*)',
    re.MULTILINE
)

def extract_images_from_md(md_path: str):
    """
    从一个 md 文件中抽取所有图片信息：
    返回 list[{"image_path": 绝对路径, "caption": caption}]
    """
    if not os.path.exists(md_path):
        # 找不到 md，返回空
        print(f"[MISS_MD] {md_path}")
        return []

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    md_dir = os.path.dirname(md_path)
    results = []

    for rel_path, caption in IMG_PATTERN.findall(text):
        abs_path = os.path.normpath(os.path.join(md_dir, rel_path))
        results.append({
            "image_path": abs_path,
            "caption": caption.strip()
        })

    return results

def build_md_path_from_id(ref_id: str):
    """
    根据 文献ID 构造 md 文件路径。
    文献ID 例如: STP1132-EB-..._9th Volume_027_chunk1
    pdf 基础名: 去掉 `_chunk数字`
    md 文件名: 文献ID + `.md`
    """
    # 去掉末尾 `_chunk数字`
    base_name = re.sub(r"_chunk\d+$", "", ref_id)
    md_filename = f"{ref_id}.md"
    md_path = os.path.join(MD_ROOT, base_name, "auto", md_filename)
    return md_path

def main():
    df = pd.read_csv(CSV_PATH)

    if "文献ID" not in df.columns:
        raise ValueError("CSV 中找不到 '文献ID' 这一列")

    image_info_cache = {}
    new_col = []

    for ref_id in df["文献ID"]:
        # 防止 NaN
        if isinstance(ref_id, float):
            ref_id = str(ref_id)

        if ref_id in image_info_cache:
            new_col.append(image_info_cache[ref_id])
            continue

        md_path = build_md_path_from_id(ref_id)
        img_list = extract_images_from_md(md_path)

        # 存成 JSON 字符串，方便后续解析
        cell_value = json.dumps(img_list, ensure_ascii=False)
        image_info_cache[ref_id] = cell_value
        new_col.append(cell_value)

    # 新增一列
    df["图片信息"] = new_col

    # 保存新的 CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"[DONE] 写入: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
