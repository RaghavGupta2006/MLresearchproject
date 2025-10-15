from .splinear import SpLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import math


# 用在GrowNN 中
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
        # print("MLP_2HL,x.shape", x.shape)
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


# 用在 tableMACCSkeysTrain 中
class MLP_Maccs(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=True, bn=True):
        super(MLP_Maccs, self).__init__()
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.5)
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
        # print("MLP_Maccs,x.shape", x.shape)
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
        model = MLP_Maccs(dim_in, args.hidden_d, args.hidden_d, args.sparse)
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


# 用于提取 表格数据 特征
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


import torchvision.models as models


# 用于 表格数据 与 分子图像 数据
class MLP_ResNet(nn.Module):
    def __init__(self, table_dim_in, table_dim_hidden, out_dim,
                 combined_dim,
                 dim_hidden1,
                 dim_hidden2,
                 sparse=True,
                 bn=True):
        super(MLP_ResNet, self).__init__()
        # 表格数据处理分支（保持原有结构）
        self.table_net = MLP(table_dim_in, table_dim_hidden, out_dim)

        # 分子图像 Resnet神经网络分支
        self.img_encoder = models.resnet18(pretrained=False)
        self.img_encoder.load_state_dict(torch.load('./resnetpth/resnet18-5c106cde.pth'), strict=False)
        # 替换最后一层全连接层以匹配特征维度
        self.img_encoder.fc = nn.Linear(self.img_encoder.fc.in_features, out_dim)
        # self.bn2 = None
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

    def forward(self, table_data, image_data, lower_f):
        # table_data 是表格数据， image_data是分子图像数据
        # 处理表格数据
        table_feat = self.table_net(table_data)
        # 处理分子图像数据
        image_feat = self.img_encoder(image_data)

        # 特征融合
        # combined = torch.cat([table_feat, graph_feat], dim=1)
        # combined = (table_feat + graph_feat) / 2
        combined = self.alpha * table_feat + (1 - self.alpha) * image_feat
        x = combined
        # print("特征融合前table_feat维度:", table_feat.shape)
        # print("特征融合前graph_feat维度:", graph_feat.shape)
        # print("特征融合后维度:", x.shape)
        # 如果不是第一个弱学习器
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            # print("我打印了self.combined_net:", x.shape)
            # if self.bn2 is None:  # 延迟初始化
            #     self.bn2 = nn.BatchNorm1d(x.shape[1]).to(x.device)
            x = self.bn2(x).to(x.device)

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
        model = MLP_ResNet(args.table_dim_in, args.table_dim_hidden, args.out_dim,
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


# 用于 表格数据 与 分子图 与 分子图像 数据
class MLP_GNNResNet(nn.Module):
    def __init__(self, table_dim_in, table_dim_hidden, gnn_input_dim,
                 out_dim, gnn_hidden, combined_dim, dim_hidden1, dim_hidden2,
                 sparse=True, bn=True):
        super(MLP_GNNResNet, self).__init__()

        attn_hidden = 32  # 适当增大容量
        self.attn_layer = nn.Sequential(
            nn.Linear(out_dim, attn_hidden * 2),
            nn.ReLU(),
            nn.Linear(attn_hidden * 2, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 3),
            nn.Softmax(dim=1)
        )

        # 表格数据处理分支
        self.table_net = MLP(table_dim_in, table_dim_hidden, out_dim)

        # 分子图数据处理分支
        self.gnn = SimpleGNN(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden,
            output_dim=out_dim
        )

        # 分子图像处理分支
        self.img_encoder = models.resnet18(pretrained=False)
        self.img_encoder.load_state_dict(
            torch.load('./resnetpth/resnet18-5c106cde.pth'), strict=False
        )
        self.img_encoder.fc = nn.Linear(self.img_encoder.fc.in_features, out_dim)

        # 图像特征增强器
        self.image_enhancer = ImageFeatureEnhancer(out_dim, out_dim * 2)
        # 上下文注意力机制（表格+分子图）
        self.context_attn = nn.Sequential(
            nn.Linear(out_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )
        # 三模态注意力机制
        self.attn_layer = nn.Sequential(
            nn.Linear(out_dim * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )

        # 特征融合参数（三分支加权融合）
        self.alpha = nn.Parameter(torch.tensor(0.34))  # 表格权重
        self.beta = nn.Parameter(torch.tensor(0.34))  # 分子图权重
        self.gamma = nn.Parameter(torch.tensor(0.33))  # 分子图像权重

        # 特征融合后处理网络
        self.bn2 = nn.BatchNorm1d(combined_dim)
        self.in_layer = SpLinear(combined_dim, dim_hidden1) if sparse else nn.Linear(combined_dim, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden1, dim_hidden2)
        self.out_layer = nn.Linear(dim_hidden2, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)

    def forward(self, table_data, graph_data, image_data, lower_f):
        # 处理三种数据类型
        table_feat = self.table_net(table_data)
        graph_feat = self.gnn(graph_data)
        image_feat = self.img_encoder(image_data)

        # 三模态加权融合（自动归一化权重）
        total_weight = self.alpha + self.beta + self.gamma
        combined = (
                (self.alpha / total_weight) * table_feat +
                (self.beta / total_weight) * graph_feat +
                (self.gamma / total_weight) * image_feat
        )

        # # 1. 先融合表格和分子图（主导特征）
        # context_feats = torch.stack([table_feat, graph_feat], dim=1)
        # context_weights = self.context_attn(torch.cat([table_feat, graph_feat], dim=1))
        # context_fused = torch.sum(context_feats * context_weights.unsqueeze(2), dim=1)
        #
        # # 2. 使用上下文信息增强图像特征
        # image_enhanced = self.image_enhancer(image_feat, context_fused)
        #
        # # 3. 三模态融合（包含增强后的图像特征）
        # all_feats = torch.stack([
        #     table_feat,
        #     graph_feat,
        #     image_enhanced
        # ], dim=1)
        #
        # # 注意力权重
        # attn_weights = self.attn_layer(
        #     torch.cat([table_feat, graph_feat, image_enhanced], dim=1))
        #
        # # 加权融合
        # combined = torch.sum(all_feats * attn_weights.unsqueeze(2), dim=1)

        # 集成先前模型的特征
        x = combined
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)

        # 后续处理网络
        out = self.lrelu(self.in_layer(x))
        out = self.dropout_layer(out)
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, stage, args):
        # 计算融合维度（考虑先前模型的特征）
        combined_dim = args.combined_dim + (args.dim_hidden2 if stage > 0 else 0)

        model = MLP_GNNResNet(
            table_dim_in=args.table_dim_in,
            table_dim_hidden=args.table_dim_hidden,
            gnn_input_dim=args.gnn_input_dim,
            out_dim=args.out_dim,
            gnn_hidden=args.gnn_hidden,
            combined_dim=combined_dim,
            dim_hidden1=args.dim_hidden1,
            dim_hidden2=args.dim_hidden2,
            sparse=args.sparse
        )

        # 权重初始化
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.manual_seed(41)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(init_weights)
        return model


class ImageFeatureEnhancer(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.enhancer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)  # 输出维度不变
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, image_feat, context_feat):
        """
        image_feat: 原始图像特征 [batch, out_dim]
        context_feat: 上下文特征（表格+分子图的融合）[batch, out_dim]
        """
        # 特征增强
        enhanced = self.enhancer(image_feat)

        # 门控机制控制增强程度
        gate_input = torch.cat([image_feat, context_feat], dim=1)
        gate_value = self.gate(gate_input)

        # 残差连接 + 门控
        return image_feat + gate_value * enhanced
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


# 用于 表格数据 与 分子图像 数据，使用Transformer提取图像特征
class MLP_Transformer(nn.Module):
    def __init__(self, table_dim_in, table_dim_hidden, out_dim,
                 combined_dim,
                 dim_hidden1,
                 dim_hidden2,
                 transformer_embed_dim=768,
                 transformer_heads=12,
                 transformer_layers=12,
                 sparse=True,
                 bn=True):
        super(MLP_Transformer, self).__init__()
        # 表格数据处理分支（保持原有结构）
        self.table_net = MLP(table_dim_in, table_dim_hidden, out_dim)

        # 分子图像 Transformer神经网络分支
        self.img_encoder = VisionTransformer(
            embed_dim=transformer_embed_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            out_dim=out_dim
        )

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

    def forward(self, table_data, image_data, lower_f):
        # table_data 是表格数据， image_data是分子图像数据
        # 处理表格数据
        table_feat = self.table_net(table_data)
        # 处理分子图像数据
        image_feat = self.img_encoder(image_data)

        # 特征融合
        combined = self.alpha * table_feat + (1 - self.alpha) * image_feat
        x = combined

        # 如果不是第一个弱学习器
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x).to(x.device)

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

        model = MLP_Transformer(args.table_dim_in, args.table_dim_hidden, args.out_dim,
                                combined_dim,
                                args.dim_hidden1,
                                args.dim_hidden2,
                                transformer_embed_dim=args.transformer_embed_dim,
                                transformer_heads=args.transformer_heads,
                                transformer_layers=args.transformer_layers,
                                sparse=args.sparse)

        # 新增：使用固定种子的权重初始化
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.manual_seed(41)  # 固定随机种子
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(init_weights)
        return model
