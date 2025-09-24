import os
import re

# 目标文件夹路径
folder_path = "/home/zuozhuo/info-extract/zr_pdfs_split_resume"

# 匹配的模式（Industry_1st 到 Industry_8th）
pattern = re.compile(r"Industry_(?:[1-8](?:st|nd|rd|th))")

# 记录删除的文件
deleted_files = []

# 遍历文件夹
for filename in os.listdir(folder_path):
    if pattern.search(filename):
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)
            deleted_files.append(filename)
        except Exception as e:
            deleted_files.append(f"删除失败: {filename}, 错误: {e}")

deleted_files[:20]  # 仅显示前20个删除的文件名字
