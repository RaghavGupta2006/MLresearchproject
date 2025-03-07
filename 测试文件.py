from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw
import pandas as pd
import numpy as np

# molecule = Chem.MolFromSmiles('Oc1ccc(cc1)C(=O)OCc2ccccc2')  # 当分子为空值时，fingerprints全为0
# fingerprints = MACCSkeys.GenMACCSKeys(molecule)
# print(fingerprints)  # result 1
# print(len(fingerprints.ToBitString()))  # result 2
# print(fingerprints.ToBitString())  # result 3
#
#
# img = Draw.MolToImage(molecule, size=(225, 225))
# # img2 = Draw.MolToImage(mol2)
# # 显示分子图
# img.show()
#
#
# file_path = "./data/processed/MemTrOC-Dataset.csv"
# data = pd.read_csv(file_path)
# data = data.head(5)
# # 统计指定列的不同值个数
# column_name = 'NAME of TrOCs'  # 替换为目标列名
# counts = data[column_name].value_counts()
#
# # 提取特征和标签
# X = data.iloc[:, 4:23].values  # 特征（19维）
# y = data.iloc[:, 23].values  # 标签
# smiles_list = data.iloc[:, 3].values  # 第3列是SMILES

# # 输出结果
# print(f"不同值及其出现次数：\n{counts}\n")
# print(f"唯一值总数：{counts.shape[0]}")
#
# total = counts.sum()
# total_rows = len(data)
# print(f"总行数：{total_rows}")
# print(f"总和是否等于总行数：{total == total_rows}")


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem import Draw


# 定义自定义Dataset类
class TabularImageDataset(Dataset):
    def __init__(self, X, smiles_list, y, image_transform=None):
        """
        Args:
            X (numpy.ndarray): 数值特征矩阵
            smiles_list (list): SMILES字符串列表
            y (numpy.ndarray): 标签数组
            image_transform (torchvision.transforms): 图像预处理变换
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.smiles_list = smiles_list
        self.y = torch.tensor(y, dtype=torch.float32)

        # 默认图像转换：调整尺寸、转为张量、归一化
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像尺寸
            transforms.ToTensor(),  # 转为PyTorch张量
            transforms.Normalize(  # RGB归一化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 获取表格数据和标签
        table_feature = self.X[idx]
        label = self.y[idx]

        # 转换SMILES为分子图像
        smiles = self.smiles_list[idx]
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"无效的SMILES: {smiles}，索引: {idx}")

        img = Draw.MolToImage(molecule, size=(224, 224))  # 生成稍大的原始图像
        img = self.image_transform(img)  # 应用预处理变换

        return table_feature, img, label


# 示例用法
if __name__ == "__main__":
    # 数据预处理（假设data已加载）
    file_path = "./data/processed/MemTrOC-Dataset.csv"
    data = pd.read_csv(file_path).head(5)  # 示例取前5条数据
    import time

    start = time.time()
    X = data.iloc[:, 4:23].values
    y = data.iloc[:, 23].values
    smiles_list = data.iloc[:, 3].values

    # 创建数据集和数据加载器
    dataset = TabularImageDataset(X, smiles_list, y)



    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    # 验证数据加载
    for table_batch, image_batch, label_batch in dataloader:
        print("表格数据尺寸:", table_batch.shape)  # 预期: (batch_size, 19)
        print("图像数据尺寸:", image_batch.shape)  # 预期: (batch_size, 3, 224, 224)
        print("标签数据尺寸:", label_batch.shape)  # 预期: (batch_size,)
        break
    # 打印 DataLoader 的第一个批次
    for table_batch, image_batch, label_batch in dataloader:
        # 打印表格特征、图数据和标签
        print("Tables Batch Shape:", table_batch.shape)
        print("Tables Batch[0]:", table_batch[0])

        print("image Batch:", image_batch.shape)
        print("image Batch[0]:", image_batch[0])

        print("Labels Batch Shape:", label_batch.shape)
        print("Labels Batch[0]:", label_batch[0])

        # 如果只需要打印一条数据，可以在这里 break
        break
    end = time.time()
    elapsed_time = end - start
    print(f"运行时间：{elapsed_time:.1f} 秒")