import torch
from torch.utils.data import Dataset
from typing import Tuple, Any
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem import Draw


# 用于创建 表格数据 + 分子图数据的 Dataset
class TableGraphDataset(Dataset):
    def __init__(self, tables, smiles, labels, create_graph_data_from_smiles):
        """
        初始化 TableGraphDataset 数据集类。

        参数:
            tables : 表格特征
            smiles (list of str): 包含SMILES字符串的列表。
            labels (list): 对应于SMILES的标签。
            create_graph_data_from_smiles (callable): 用于将SMILES转换成图数据格式的函数。
        """
        self.tables = tables
        self.smiles = smiles
        self.labels = labels
        self.create_graph_data_from_smiles = create_graph_data_from_smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx) -> Tuple[Any, Any, Any]:
        table = self.tables[idx]
        smile = self.smiles[idx]
        label = self.labels[idx]

        # 根据smiles字符串生成图数据
        graph_data = self.create_graph_data_from_smiles(str(smile), label)

        return table, graph_data, label


# 用于创建 表格数据 + 分子图像数据的 Dataset
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
