from .splinear import SpLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import math


class MLP_2HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=True, bn=True):
        super(MLP_2HL, self).__init__()
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden1, dim_hidden2)
        self.out_layer = nn.Linear(dim_hidden2, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)
        self.bn2 = nn.BatchNorm1d(dim_in)

    def forward(self, x, lower_f):
        # 第一个弱学习器输入不是 当前输入 + 上一个学习器倒数第二层
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        # print(x.shape)
        out = self.lrelu(self.in_layer(x))
        out = self.dropout_layer(out)
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, stage, args):
        if stage == 0:
            dim_in = args.feat_d
        else:
            dim_in = args.feat_d + args.hidden_d
        model = MLP_2HL(dim_in, args.hidden_d, args.hidden_d, args.sparse)
        return model


# 用于特征融合
class SelfAttentionFusion(nn.Module):
    def __init__(self, d_model=128, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead)

        # 可选的层归一化和前馈层
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, table_feat, graph_feat):
        # 输入形状: table_feat [64, 128], graph_feat [64, 128]

        # 将特征拼接为序列 [2, 64, 128]
        combined = torch.stack([table_feat, graph_feat], dim=0)

        # 自注意力计算 (自动处理batch维度)
        attn_output, _ = self.self_attn(combined, combined, combined)

        # 残差连接 + 层归一化
        combined = self.norm(combined + attn_output)

        # 前馈网络 (作用于每个位置)
        output = self.ffn(combined)  # [2, 64, 128]

        # 合并策略：取平均 (也可改为其他方式)
        fused_output = torch.mean(output, dim=0)  # [64, 128]

        return fused_output


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=True, bn=True):
        super(MLP, self).__init__()
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden1, dim_hidden2)
        self.out_layer = nn.Linear(dim_hidden2, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)
        self.bn2 = nn.BatchNorm1d(dim_in)

    def forward(self, x):
        # print(x.shape)
        out = self.lrelu(self.in_layer(x))
        out = self.dropout_layer(out)
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out


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


# 用于表格数据与图数据
class MLP_GNN(nn.Module):
    def __init__(self, table_dim_in, table_dim_hidden, gnn_input_dim, out_dim, gnn_hidden,
                 combined_dim,
                 dim_hidden1,
                 dim_hidden2,
                 sparse=True,
                 bn=True):
        super(MLP_GNN, self).__init__()
        # 表格数据处理分支（保持原有结构）
        self.table_net = MLP(table_dim_in, table_dim_hidden, out_dim)

        # 图神经网络分支
        self.gnn = SimpleGNN(input_dim=gnn_input_dim,  # 根据原子特征维度调整
                             hidden_dim=gnn_hidden,
                             output_dim=out_dim)
        #
        self.bn2 = nn.BatchNorm1d(combined_dim)
        # 特征融合层
        # 加权融合参数（可学习的权重）
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始权重为0.5
        # 下面是特征融合之后走的网络
        self.in_layer = SpLinear(combined_dim, dim_hidden1) if sparse else nn.Linear(combined_dim, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden1, dim_hidden2)
        self.out_layer = nn.Linear(dim_hidden2, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)

    def forward(self, table_data, graph_data, lower_f):
        # table_data 是表格数据， graph_data是图数据
        # 处理表格数据
        table_feat = self.table_net(table_data)
        # 处理图数据
        graph_feat = self.gnn(graph_data)

        # 特征融合
        # combined = torch.cat([table_feat, graph_feat], dim=1)
        # combined = (table_feat + graph_feat) / 2
        combined = self.alpha * table_feat + (1 - self.alpha) * graph_feat
        x = combined
        # print("特征融合前table_feat维度:", table_feat.shape)
        # print("特征融合前graph_feat维度:", graph_feat.shape)
        # print("特征融合后维度:", x.shape)
        # 如果不是第一个弱学习器
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            # print("我打印了self.combined_net:", x.shape)
            x = self.bn2(x)

        # print("self.combined_net:", x.shape)
        out = self.lrelu(self.in_layer(x))
        out = self.dropout_layer(out)
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, stage, args):
        if stage == 0:
            combined_dim = args.combined_dim
        else:
            # 如果不是第一个弱学习器，则表格特征与图特征融合后 特征维度就是 原融合后维度 + 倒数第二层的输出维度
            combined_dim = args.combined_dim + args.dim_hidden2
        # print("stage:", stage, "  combined_dim:", combined_dim)
        model = MLP_GNN(args.table_dim_in, args.table_dim_hidden, args.gnn_input_dim, args.out_dim, args.gnn_hidden,
                        combined_dim,
                        args.dim_hidden1,
                        args.dim_hidden2, args.sparse)

        # 新增：使用固定种子的权重初始化
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.manual_seed(41)  # 固定随机种子
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(init_weights)
        return model


class MLP_3HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_3HL, self).__init__()
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.0)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden2, dim_hidden1)
        self.out_layer = nn.Linear(dim_hidden1, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)
        self.bn2 = nn.BatchNorm1d(dim_in)
        # print('Batch normalization is processed!')

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        out = self.lrelu(self.in_layer(x))
        out = self.bn(out)
        out = self.lrelu(self.hidden_layer(out))
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, stage, opt):
        if stage == 0:
            dim_in = opt.feat_d
        else:
            dim_in = opt.feat_d + opt.hidden_d
        model = MLP_3HL(dim_in, opt.hidden_d, opt.hidden_d, opt.sparse)
        return model
