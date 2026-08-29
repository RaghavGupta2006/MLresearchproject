"""
Utility script to convert raw Excel (.xlsx) datasets into CSV format.
"""
import os
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
xlsx_file_path = os.path.join(base_dir, 'data', 'processed', 'MemTrOC-Dataset.xlsx')
csv_file_path = os.path.join(base_dir, 'data', 'processed', 'MemTrOC-Dataset.csv')

if os.path.exists(xlsx_file_path):
    excel_file = pd.ExcelFile(xlsx_file_path)
    sheet_name = excel_file.sheet_names[0]
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df.to_csv(csv_file_path, index=False)
    print(f"Sheet '{sheet_name}' successfully converted to CSV: {csv_file_path}")
else:
    print(f"File not found: {xlsx_file_path}")
