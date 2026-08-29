from .splinear import SpLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool, global_max_pool, global_add_pool
import math
import torchvision.models as models


# Used in GrowNN (Pure Tabular Baseline)
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
        # If not the first weak learner, concatenate current input with prior stage hidden features
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
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


# Used in tableMACCSkeysTrain (Tabular + Fingerprints Baseline)
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
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
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


# Self-Attention Feature Fusion Module
class SelfAttentionFusion(nn.Module):
    def __init__(self, d_model=128, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, table_feat, graph_feat):
        # Input shapes: table_feat [batch_size, d_model], graph_feat [batch_size, d_model]
        combined = torch.stack([table_feat, graph_feat], dim=0)

        # Multi-head attention
        attn_output, _ = self.self_attn(combined, combined, combined)

        # Residual connection + LayerNorm
        combined = self.norm(combined + attn_output)

        # Feed-forward projection
        output = self.ffn(combined)

        # Mean pooling aggregation
        fused_output = torch.mean(output, dim=0)
        return fused_output


# Tabular Descriptor Feature Extractor
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
        out = self.lrelu(self.in_layer(x))
        out = self.dropout_layer(out)
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out


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


# Advanced Molecular Graph Neural Network - GINEConv + 3D Bond Features + Multi-Scale Readout
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

        # Multi-Scale Readout projection: concatenate mean_pool + max_pool + sum_pool (3 * hidden_dim) -> output_dim
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


# Primary Multimodal Weak Learner: Tabular Descriptors + Molecular Graph (GINE / GCN)
class MLP_GNN(nn.Module):
    def __init__(self, table_dim_in, table_dim_hidden, gnn_input_dim, out_dim, gnn_hidden,
                 combined_dim,
                 dim_hidden1,
                 dim_hidden2,
                 sparse=True,
                 bn=True,
                 gnn_type='gine',
                 edge_dim=3):
        super(MLP_GNN, self).__init__()
        # Tabular descriptor branch
        self.table_net = MLP(table_dim_in, table_dim_hidden, out_dim)

        # Graph Neural Network branch
        self.gnn_type = gnn_type
        if gnn_type == 'gine':
            self.gnn = GINE_MolecularGNN(input_dim=gnn_input_dim,
                                         edge_dim=edge_dim,
                                         hidden_dim=gnn_hidden,
                                         output_dim=out_dim)
        else:
            self.gnn = SimpleGNN(input_dim=gnn_input_dim,
                                 hidden_dim=gnn_hidden,
                                 output_dim=out_dim)

        self.bn2 = nn.BatchNorm1d(combined_dim)
        # Learnable convex fusion weighting
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Downstream regression prediction head
        self.in_layer = SpLinear(combined_dim, dim_hidden1) if sparse else nn.Linear(combined_dim, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden1, dim_hidden2)
        self.out_layer = nn.Linear(dim_hidden2, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)

    def forward(self, table_data, graph_data, lower_f):
        # Extract features
        table_feat = self.table_net(table_data)
        graph_feat = self.gnn(graph_data)

        # Convex feature fusion
        combined = self.alpha * table_feat + (1 - self.alpha) * graph_feat
        x = combined

        # Concatenate prior stage hidden representation
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)

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
            combined_dim = args.combined_dim + args.dim_hidden2

        gnn_type = getattr(args, 'gnn_type', 'gine')
        edge_dim = getattr(args, 'edge_dim', 3)

        model = MLP_GNN(args.table_dim_in, args.table_dim_hidden, args.gnn_input_dim, args.out_dim, args.gnn_hidden,
                        combined_dim,
                        args.dim_hidden1,
                        args.dim_hidden2,
                        args.sparse,
                        gnn_type=gnn_type,
                        edge_dim=edge_dim)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.manual_seed(41)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(init_weights)
        return model


# Multimodal Weak Learner: Tabular Descriptors + 2D Molecular Images (ResNet)
class MLP_ResNet(nn.Module):
    def __init__(self, table_dim_in, table_dim_hidden, out_dim,
                 combined_dim,
                 dim_hidden1,
                 dim_hidden2,
                 sparse=True,
                 bn=True):
        super(MLP_ResNet, self).__init__()
        self.table_net = MLP(table_dim_in, table_dim_hidden, out_dim)

        self.img_encoder = models.resnet18(pretrained=False)
        # Adapt final fully connected layer
        self.img_encoder.fc = nn.Linear(self.img_encoder.fc.in_features, out_dim)
        self.bn2 = nn.BatchNorm1d(combined_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.in_layer = SpLinear(combined_dim, dim_hidden1) if sparse else nn.Linear(combined_dim, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden1, dim_hidden2)
        self.out_layer = nn.Linear(dim_hidden2, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)

    def forward(self, table_data, image_data, lower_f):
        table_feat = self.table_net(table_data)
        image_feat = self.img_encoder(image_data)

        combined = self.alpha * table_feat + (1 - self.alpha) * image_feat
        x = combined

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
            combined_dim = args.combined_dim + args.dim_hidden2

        model = MLP_ResNet(args.table_dim_in, args.table_dim_hidden, args.out_dim,
                           combined_dim,
                           args.dim_hidden1,
                           args.dim_hidden2, args.sparse)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.manual_seed(41)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(init_weights)
        return model


# Tri-Modal Weak Learner: Tabular + Molecular Graph + Molecular Image
class MLP_GNNResNet(nn.Module):
    def __init__(self, table_dim_in, table_dim_hidden, gnn_input_dim,
                 out_dim, gnn_hidden, combined_dim, dim_hidden1, dim_hidden2,
                 sparse=True, bn=True):
        super(MLP_GNNResNet, self).__init__()

        self.table_net = MLP(table_dim_in, table_dim_hidden, out_dim)
        self.gnn = SimpleGNN(input_dim=gnn_input_dim, hidden_dim=gnn_hidden, output_dim=out_dim)
        self.img_encoder = models.resnet18(pretrained=False)
        self.img_encoder.fc = nn.Linear(self.img_encoder.fc.in_features, out_dim)

        self.alpha = nn.Parameter(torch.tensor(0.34))
        self.beta = nn.Parameter(torch.tensor(0.34))
        self.gamma = nn.Parameter(torch.tensor(0.33))

        self.bn2 = nn.BatchNorm1d(combined_dim)
        self.in_layer = SpLinear(combined_dim, dim_hidden1) if sparse else nn.Linear(combined_dim, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden1, dim_hidden2)
        self.out_layer = nn.Linear(dim_hidden2, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)

    def forward(self, table_data, graph_data, image_data, lower_f):
        table_feat = self.table_net(table_data)
        graph_feat = self.gnn(graph_data)
        image_feat = self.img_encoder(image_data)

        total_weight = self.alpha + self.beta + self.gamma
        combined = (
                (self.alpha / total_weight) * table_feat +
                (self.beta / total_weight) * graph_feat +
                (self.gamma / total_weight) * image_feat
        )

        x = combined
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)

        out = self.lrelu(self.in_layer(x))
        out = self.dropout_layer(out)
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, stage, args):
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

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.manual_seed(41)
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
