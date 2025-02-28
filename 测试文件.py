from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw
import pandas as pd
import numpy as np

molecule = Chem.MolFromSmiles('Oc1ccc(cc1)C(=O)OCc2ccccc2')  # 当分子为空值时，fingerprints全为0
fingerprints = MACCSkeys.GenMACCSKeys(molecule)
print(fingerprints)  # result 1
print(len(fingerprints.ToBitString()))  # result 2
print(fingerprints.ToBitString())  # result 3


# img = Draw.MolToImage(molecule, size=(225, 225))
# # img2 = Draw.MolToImage(mol2)
# # 显示分子图
# img.show()





print(smiles_to_maccs('Oc1ccc(cc1)C(=O)OCc2ccccc2'))
file_path = "./data/processed/MemTrOC-Dataset.csv"
data = pd.read_csv(file_path)
data = data.head(5)
# 统计指定列的不同值个数
column_name = 'NAME of TrOCs'  # 替换为目标列名
counts = data[column_name].value_counts()

# 提取特征和标签
X = data.iloc[:, 4:23].values  # 特征（19维）
y = data.iloc[:, 23].values  # 标签
smiles_list = data.iloc[:, 3].values  # 第3列是SMILES

# # 输出结果
# print(f"不同值及其出现次数：\n{counts}\n")
# print(f"唯一值总数：{counts.shape[0]}")
#
# total = counts.sum()
# total_rows = len(data)
# print(f"总行数：{total_rows}")
# print(f"总和是否等于总行数：{total == total_rows}")
