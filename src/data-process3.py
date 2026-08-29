import pandas as pd

# CSV
file_path = '../data/processed/final_data.xlsx'  # CSV
df = pd.read_excel(file_path)

# Note: processed parameter
column_name = 'Type of Membranes'  # Note: processed parameter

# Value
unique_values_count = df[column_name].nunique()

print(f"  '{column_name}'  Value : {unique_values_count}")

# Value
value_counts = df[column_name].value_counts()

print(f"  '{column_name}'  Value :")
print(value_counts)
print("---------------------------------------------------")
# Note: processed parameter
print("CSV :")
print(df.info())
print("---------------------------------------------------")
# Note: processed parameter
# print("\n :")
# print(df.describe(include='all'))
print("---------------------------------------------------")
# Name、 Value
print("\n Name、 Value :")
for column in df.columns:
    nan_count = df[column].isna().sum()
    print(f" : {column},  : {df[column].dtype},  Value : {nan_count}")

print("-------------------------------------------------------")
# 'Pore radius (nm)'   Value Value
mean_value = df['Pore radius (nm)'].mean()  # Value
df['Pore radius (nm)'].fillna(mean_value, inplace=True)  # Value Value
# Value
df = df.dropna(subset=['pKa1 ', 'Molecular radius (nm)', 'log Kow'])
# Name、 Value
print("\n Name、 Value :")
for column in df.columns:
    nan_count = df[column].isna().sum()
    print(f" : {column},  : {df[column].dtype},  Value : {nan_count}")


# Note: processed parameter
columns_to_describe = ['Pore radius (nm)', 'Molecular charge', 'Charge product']

# DataFrame
df_to_describe = df[columns_to_describe]

# DataFrame   describe()
description = df_to_describe.describe()

print(description)


df.to_excel('../data/processed/MemTrOC-Dataset.xlsx', index=False)