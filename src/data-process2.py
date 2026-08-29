import pandas as pd

# CSV
file_path = '../data/processed/Membrane-pollutants_updated.xlsx'  # CSV
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
print("\n :")
print(df.describe(include='all'))
print("---------------------------------------------------")
# Name、 Value
print("\n Name、 Value :")
for column in df.columns:
    nan_count = df[column].isna().sum()
    print(f" : {column},  : {df[column].dtype},  Value : {nan_count}")

print("-------------------------------------------------------")
# #  Value
df_drop = df.drop(columns=['pKa2', 'ɸ', 'Cross-flow velocity   (cm·s-1)', 'Data group'])

df_drop.to_excel('../data/processed/final_data.xlsx', index=False)

# Note: processed parameter
columns_to_describe = ['Pore radius (nm)', 'MWCO (Da)']

# DataFrame
df_to_describe = df[columns_to_describe]

# DataFrame   describe()
description = df_to_describe.describe()

print(description)