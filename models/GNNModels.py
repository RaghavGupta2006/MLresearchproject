import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool, global_max_pool, global_add_pool
import math


# Self-Attention Module for Feature Fusion
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


# Molecular Graph Neural Network - Standard GCN Backbone
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
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        return x


# Advanced Molecular Graph Neural Network - GINEConv + 3D Bond Features + Multi-Scale Readout (Mean + Max + Sum)
class GINE_MolecularGNN(nn.Module):
    def __init__(self, input_dim=9, edge_dim=3, hidden_dim=128, output_dim=128, dropout=0.2):
        super(GINE_MolecularGNN, self).__init__()
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None

        # GINE Layer 1
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINEConv(nn1, edge_dim=hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # GINE Layer 2
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(nn2, edge_dim=hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Multi-Scale Readout: concatenate mean_pool + max_pool + sum_pool (3 * hidden_dim) -> output_dim
        self.readout_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)

        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr.float())
        else:
            edge_attr = None

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.gelu(x)

        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        pool_mean = global_mean_pool(x, batch)
        pool_max = global_max_pool(x, batch)
        pool_sum = global_add_pool(x, batch)

        pooled = torch.cat([pool_mean, pool_max, pool_sum], dim=1)
        return self.readout_proj(pooled)


"""
Gradient Boosted Neural Network + Graph Neural Network Module
"""


class DynamicNetGNNModel(nn.Module):
    def __init__(self, table_input_dim, gnn_input_dim, table_hidden=128, gnn_hidden=128, combined_hidden=256):
        super().__init__()
        # Tabular descriptor processing branch
        from .weaklearner import MLP_2HL
        self.table_net = MLP_2HL(table_input_dim, table_hidden, table_hidden)

        # Graph Neural Network branch
        self.gnn = SimpleGNN(input_dim=gnn_input_dim,
                             hidden_dim=gnn_hidden,
                             output_dim=gnn_hidden)

        # Feature fusion head
        self.combined_net = nn.Sequential(
            nn.Linear(table_hidden + gnn_hidden, combined_hidden),
            nn.ReLU(),
            nn.Linear(combined_hidden, 1)
        )

    def forward(self, x_table, graph_data):
        # Process tabular features
        table_feat, _ = self.table_net(x_table, None)

        # Process graph topology
        graph_feat = self.gnn(graph_data)

        # Feature fusion and prediction
        combined = torch.cat([table_feat, graph_feat], dim=1)
        output = self.combined_net(combined)
        return output
