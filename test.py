import os
import pandas as pd

base_dir = "/home/zuozhuo/info-extract/zr_output"

# 1. 统计子文件夹数量
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"路径不存在: {base_dir}")

subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
num_subfolders = len(subfolders)

# 2. 统计所有 final.csv 文件的总行数（不算表头），并保存路径
total_rows = 0
final_csv_files = []

for subfolder in subfolders:
    for file in os.listdir(subfolder):
        if file.endswith("final.csv"):
            file_path = os.path.join(subfolder, file)
            final_csv_files.append(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    row_count = sum(1 for _ in f) - 1  # 去掉表头
                    total_rows += max(row_count, 0)
            except Exception as e:
                print(f"读取文件失败: {file_path}, 错误: {e}")

print(f"子文件夹数量: {num_subfolders}")
print(f"找到的 final.csv 文件数量: {len(final_csv_files)}")
print(f"所有 final.csv 文件的总行数（不含表头）: {total_rows}")
print(f"平均每篇提取条数：{total_rows / num_subfolders if num_subfolders > 0 else 0:.2f}")

# 3. 合并所有 final.csv
if final_csv_files:
    dfs = []
    for f in final_csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8")
            dfs.append(df)
        except Exception as e:
            print(f"读取 {f} 失败: {e}")
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(base_dir, "final.csv")
        merged_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ 已合并所有 final.csv，输出路径: {output_path}")
    else:
        print("⚠️ 没有成功读取到任何 final.csv，无法合并")
else:
    print("⚠️ 没有找到任何 final.csv 文件")
