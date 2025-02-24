import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from dataset.dataset import TableGraphDataset
from src.utils.smiles2graph import create_graph_data_from_smiles
import torch

# 数据路径
file_path = "./data/processed/MemTrOC-Dataset.csv"
data = pd.read_csv(file_path)
data = data.head(10)
# 提取特征和标签
X = data.iloc[:, 4:23].values  # 特征（19维）
y = data.iloc[:, 23].values  # 标签

smiles_list = data.iloc[:, 3].values  # 第3列是SMILES

# 先划分数据集再进行归一化（关键修改！）
X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles_list, test_size=0.1, random_state=41)
X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(X_train, y_train, smiles_train, test_size=0.2 / 0.9, random_state=41)
# 创建数据集
train_dataset = TableGraphDataset(X_train, smiles_train, y_train, create_graph_data_from_smiles)
val_dataset = TableGraphDataset(X_val, smiles_val, y_val, create_graph_data_from_smiles)
test_dataset = TableGraphDataset(X_test, smiles_test, y_test, create_graph_data_from_smiles)

# 打印数据集的大小
print(f"训练集大小：{len(train_dataset)}")

torch.set_printoptions(threshold=torch.inf)
# 查看单个样本
for i in range(5):  # 例如，查看前5个样本
    table, graph_data, label = train_dataset[i]
    print(f"Sample {i+1}:")
    print("  Table:", table)
    print("--------------------------------------------")
    print("  Graph data:", graph_data)
    print("graph_data.x:", graph_data.x)
    print("graph_data.edge_index:", graph_data.edge_index)
    print("graph_data.y:", graph_data.y)
    print("--------------------------------------------")
    print("  Label:", label)
    print("\n---\n")


