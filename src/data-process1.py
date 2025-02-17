import pandas as pd

# 读取CSV文件
file_path = './data/original_data.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 指定要查询的列名
column_name = 'Type of Membranes'  # 替换为你想要查询的列名

# 获取该列的不同值及其个数
unique_values_count = df[column_name].nunique()

print(f"列 '{column_name}' 中不同值的个数是: {unique_values_count}")

# 获取该列的不同值及其各自的数量
value_counts = df[column_name].value_counts()

print(f"列 '{column_name}' 中不同值的数量如下:")
print(value_counts)
print("---------------------------------------------------")
# 输出各个列的信息
print("CSV文件中的各个列信息如下:")
print(df.info())
print("---------------------------------------------------")
# 输出各个列的基本统计信息
print("\n各个列的基本统计信息:")
print(df.describe(include='all'))
print("---------------------------------------------------")
# 输出各个列的名称、数据类型和空值数量
print("\n各个列的名称、数据类型和空值数量:")
for column in df.columns:
    nan_count = df[column].isna().sum()
    print(f"列名: {column}, 数据类型: {df[column].dtype}, 空值数量: {nan_count}")


# # 读取 original_data.xlsx 文件的第一个 sheet
# original_data = pd.read_excel('./data/original_data.xlsx', sheet_name=0)
#
# # 读取 Membrane-pollutants.xlsx 文件的第一个 sheet
# membrane_pollutants = pd.read_excel('./data/Membrane-pollutants.xlsx', sheet_name=0)
#
# # 根据 'NAME of TrOCs' 列进行合并
# merged_data = membrane_pollutants.merge(original_data[['NAME of TrOCs', 'SMILEs']], on='NAME of TrOCs', how='left')
#
# # 将合并后的数据保存回 Membrane-pollutants.xlsx 文件
# merged_data.to_excel('./data/Membrane-pollutants_updated.xlsx', index=False)
