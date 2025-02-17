# 将xlsx转为csv
import pandas as pd


# 指定你的xlsx文件路径和输出csv文件路径
xlsx_file_path = '../../data/processed/MemTrOC-Dataset.xlsx'
csv_file_path = '../../data/processed/MemTrOC-Dataset.csv'

# 读取Excel文件
file_path = xlsx_file_path  # 替换为你的Excel文件路径
excel_file = pd.ExcelFile(file_path)

# 获取第三个sheet的名字（索引从0开始，所以第三个sheet的索引是2）
sheet_name = excel_file.sheet_names[0]

# 读取第三个sheet的数据
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# 将数据写入CSV文件
csv_file_path = csv_file_path  # 替换为你想要保存的CSV文件路径
df.to_csv(csv_file_path, index=False)

print(f"工作表 '{sheet_name}' 已成功转换为 CSV 文件: {csv_file_path}")




