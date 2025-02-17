import pandas as pd

data = pd.read_csv("prediction of 204 membranes.csv")
df_A = data.drop_duplicates(subset=['monomer_A'])
df_B = data.drop_duplicates(subset=['monomer_B'])
select_col_A = ['monomer_A', 'fullname_A', 'smiles_A']
select_col_B = ['monomer_B', 'fullname_B', 'smiles_B']
df_uniqueA = df_A[select_col_A]
df_uniqueB = df_B[select_col_B]

dictionary_A = df_uniqueA.set_index('monomer_A')['smiles_A'].to_dict()  # 生成单体与SMILES对应的字典
dictionary_B = df_uniqueB.set_index('monomer_B')['smiles_B'].to_dict()

print(dictionary_A)

print("----------------------")
# df = pd.read_csv("no_solvent_solute_osn.csv")
# df['smiles_A'] = df['monomer_A'].apply(lambda x: dictionary_A[x])
#
# # 获取最后一列的列名和数据
# last_column_name = df.columns[-1]
# last_column_data = df[last_column_name]
# # 删除原来的最后一列
# df = df.drop(columns=[last_column_name])
# # 在第二列的位置插入最后一列
# df.insert(1, last_column_name, last_column_data)
#
# df['smiles_B'] = df['monomer_B'].apply(lambda x: dictionary_B[x])  # 在最后一列生成了smiles_B
# # 获取最后一列的列名和数据
# last_column_name = df.columns[-1]
# last_column_data = df[last_column_name]
# # 删除原来的最后一列
# df = df.drop(columns=[last_column_name])
# # 在第4列的位置插入最后一列
# df.insert(3, last_column_name, last_column_data)

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

df = pd.read_csv("last_data.csv")
# 定义 SMILES 字符串
smiles = 'Nc1cccc(N)c1'
mpd = 'Nc1cccc(N)c1'
tmc = 'O=C(Cl)c1cc(C(=O)Cl)cc(C(=O)Cl)c1'
# 读取化合物分子
m = Chem.MolFromSmiles('Nc1cccc(N)c1')


def getTPSA(m):
    return Descriptors.TPSA(Chem.MolFromSmiles(m))


def getMolLogP(m):
    return Descriptors.MolLogP(Chem.MolFromSmiles(m))


def getMolWt(m):
    return Descriptors.MolWt(Chem.MolFromSmiles(m))


def getqed(m):
    return Descriptors.qed(Chem.MolFromSmiles(m))


def getBertzCT(m):
    return Descriptors.BertzCT(Chem.MolFromSmiles(m))


def getNumHAcceptors(m):
    return Descriptors.NumHAcceptors(Chem.MolFromSmiles(m))


def getNumHDonors(m):
    return Descriptors.NumHDonors(Chem.MolFromSmiles(m))


# print(Descriptors.TPSA(m))  # 分子的极性表面积（Topological Polar Surface Area）
# print(Descriptors.MolLogP(m))  # 计算分子的极性表面积。
# print(Descriptors.MolWt(m))  # 算分子的相对分子质量。
# print(Descriptors.qed(m))  # 表示量子效应设计得分 它基于药物样性的多个关键特征，如溶解度、脂水分配系数、生物可利用性等进行计算。较高的qed值表示化合物更有可能具有良好的药物样性。
#
# print(Descriptors.BertzCT(m))  # Bertz_ct是一种分子描述符，用于衡量分子的环结构复杂度
# # 计算分子中可供氢键给体位点数
# mol = Chem.MolFromSmiles(smiles)
# print(rdMolDescriptors.CalcNumHBA(mol))  # 分子中可供氢键给体位点数（Hydrogen Bond Acceptor，HBA）
# print(Descriptors.NumHAcceptors(m))
#
# print(rdMolDescriptors.CalcNumHBD(mol))
# print(Descriptors.NumHDonors(m))

# 根据monomer_A的SMILES与rdkit计算出一些分子描述符
temp = []
temp = df['smiles_A'].apply(lambda x: getTPSA(x))
df.insert(4, "A_TPSA", temp)
temp = df['smiles_A'].apply(lambda x: getMolLogP(x))
df.insert(5, "A_MolLogP", temp)
temp = df['smiles_A'].apply(lambda x: getMolWt(x))
df.insert(6, "A_MolWt", temp)
temp = df['smiles_A'].apply(lambda x: getqed(x))
df.insert(7, "A_qed", temp)
temp = df['smiles_A'].apply(lambda x: getBertzCT(x))
df.insert(8, "A_BertzCT", temp)
temp = df['smiles_A'].apply(lambda x: getNumHAcceptors(x))
df.insert(9, "A_NumHAcceptors", temp)
temp = df['smiles_A'].apply(lambda x: getNumHDonors(x))
df.insert(10, "A_NumHDonors", temp)
# 根据monomer_B的SMILES与rdkit计算出一些分子描述符
temp = []
temp = df['smiles_B'].apply(lambda x: getTPSA(x))
df.insert(11, "B_TPSA", temp)
temp = df['smiles_B'].apply(lambda x: getMolLogP(x))
df.insert(12, "B_MolLogP", temp)
temp = df['smiles_B'].apply(lambda x: getMolWt(x))
df.insert(13, "B_MolWt", temp)
temp = df['smiles_B'].apply(lambda x: getqed(x))
df.insert(14, "B_qed", temp)
temp = df['smiles_B'].apply(lambda x: getBertzCT(x))
df.insert(15, "B_BertzCT", temp)
temp = df['smiles_B'].apply(lambda x: getNumHAcceptors(x))
df.insert(16, "B_NumHAcceptors", temp)
temp = df['smiles_B'].apply(lambda x: getNumHDonors(x))
df.insert(17, "B_NumHDonors", temp)
# print(df)
# df.to_csv("temp.csv", index=None)
