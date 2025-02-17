import pandas as pd

# 读取CSV文件
file_path = '../data/processed/final_data.xlsx'  # 替换为你的CSV文件路径
df = pd.read_excel(file_path)

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
# print("\n各个列的基本统计信息:")
# print(df.describe(include='all'))
print("---------------------------------------------------")
# 输出各个列的名称、数据类型和空值数量
print("\n各个列的名称、数据类型和空值数量:")
for column in df.columns:
    nan_count = df[column].isna().sum()
    print(f"列名: {column}, 数据类型: {df[column].dtype}, 空值数量: {nan_count}")

print("-------------------------------------------------------")
# 'Pore radius (nm)'  对该列的空值使用均值进行填充
mean_value = df['Pore radius (nm)'].mean()  # 计算列的均值
df['Pore radius (nm)'].fillna(mean_value, inplace=True)  # 使用均值填充空值
# 删除指定列为空值的行
df = df.dropna(subset=['pKa1 ', 'Molecular radius (nm)', 'log Kow'])
# 输出各个列的名称、数据类型和空值数量
print("\n各个列的名称、数据类型和空值数量:")
for column in df.columns:
    nan_count = df[column].isna().sum()
    print(f"列名: {column}, 数据类型: {df[column].dtype}, 空值数量: {nan_count}")


# 指定你想要描述的列名列表
columns_to_describe = ['Pore radius (nm)', 'Molecular charge', 'Charge product']

# 选择这些列来创建一个新的 DataFrame
df_to_describe = df[columns_to_describe]

# 对新 DataFrame 调用 describe() 方法
description = df_to_describe.describe()

print(description)


df.to_excel('../data/processed/MemTrOC-Dataset.xlsx', index=False)