import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import math


# 自注意力模块（仅用于特征融合）
class SingleHeadAttention(nn.Module):

    def __init__(self, num_hidden_k):
        super(SingleHeadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = torch.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        result = torch.bmm(attn, value)

        return result, attn


# 图神经网络
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, data.batch)
        return x


"""
梯度提升神经网络 + 图神经网络
"""


class DynamicNetGNNModel(nn.Module):
    def __init__(self, table_input_dim, gnn_input_dim, table_hidden=128, gnn_hidden=128, combined_hidden=256):
        super().__init__()
        # 表格数据处理分支（保持原有结构）
        self.table_net = MLP_2HL(table_input_dim, table_hidden)

        # 图神经网络分支
        self.gnn = GNN(input_dim=5,  # 根据原子特征维度调整
                       hidden_dim=gnn_hidden,
                       output_dim=gnn_hidden)

        # 特征融合层
        self.combined_net = nn.Sequential(
            nn.Linear(table_hidden + gnn_hidden, combined_hidden),
            nn.ReLU(),
            nn.Linear(combined_hidden, 1)
        )

    def forward(self, x_table, graph_data):
        # 处理表格数据
        table_feat = self.table_net(x_table)

        # 处理图数据
        graph_feat = self.gnn(graph_data.x,
                              graph_data.edge_index,
                              graph_data.batch)

        # 特征融合
        combined = torch.cat([table_feat, graph_feat], dim=1)
        output = self.combined_net(combined)
        return output
