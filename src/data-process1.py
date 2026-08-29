import pandas as pd

# CSV
file_path = './data/original_data.csv'  # CSV
df = pd.read_csv(file_path)

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


# #   original_data.xlsx   sheet
# original_data = pd.read_excel('./data/original_data.xlsx', sheet_name=0)
# # #   Membrane-pollutants.xlsx   sheet
# membrane_pollutants = pd.read_excel('./data/Membrane-pollutants.xlsx', sheet_name=0)
# # #   'NAME of TrOCs'
# merged_data = membrane_pollutants.merge(original_data[['NAME of TrOCs', 'SMILEs']], on='NAME of TrOCs', how='left')
# # #   Membrane-pollutants.xlsx
# merged_data.to_excel('./data/Membrane-pollutants_updated.xlsx', index=False)
