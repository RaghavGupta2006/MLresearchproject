import torch
from torch.utils.data import Dataset
from typing import Tuple, Any


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
