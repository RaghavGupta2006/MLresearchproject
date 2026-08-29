"""
PhysiChem-GT: Physics-Informed Chemical Graph Transformer
=========================================================
A completely new end-to-end model architecture that REPLACES the DynamicNet
gradient boosting framework from the base paper (Xiao et al., 2026).

Key differences from base paper:
  - No gradient boosting / no DynamicNet / no sequential weak learners
  - GATv2Conv with edge features (attention-based message passing)
  - Virtual Node for global molecular context
  - Cross-Modal Attention Fusion (dynamic per-sample weighting)
  - Physics-Constrained Composite Loss
  - MC-Dropout Uncertainty Quantification
  - Single end-to-end differentiable model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, GINEConv, SAGEConv,
    global_mean_pool, global_max_pool, global_add_pool
)
import math


# ============================================================================
# Component 1: Molecular Graph Encoder (GATv2 + Virtual Node)
# ============================================================================

class GATv2MolecularEncoder(nn.Module):
    """
    Attention-based molecular graph encoder using GATv2Conv with edge features.
    Each bond gets an attention score — interpretable bond importance.
    """
    def __init__(self, node_dim=9, edge_dim=3, hidden_dim=128, num_layers=2,
                 heads=4, dropout=0.2, use_virtual_node=True):
        super().__init__()
        self.num_layers = num_layers
        self.use_virtual_node = use_virtual_node
        self.hidden_dim = hidden_dim

        # Node feature projection
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # Edge feature projection
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # GATv2 layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            self.convs.append(
                GATv2Conv(in_dim, hidden_dim // heads, heads=heads,
                          edge_dim=hidden_dim, dropout=dropout, concat=True)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Virtual node embedding (learnable)
        if use_virtual_node:
            self.virtual_node_embedding = nn.Embedding(1, hidden_dim)
            self.virtual_node_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # Multi-scale readout projection
        self.readout_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Encode node and edge features
        x = self.node_encoder(x.float())

        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr.float())

        # Virtual node initialization
        if self.use_virtual_node:
            num_graphs = batch.max().item() + 1
            vn_embed = self.virtual_node_embedding(
                torch.zeros(num_graphs, dtype=torch.long, device=x.device)
            )

        # Message passing layers
        for i in range(self.num_layers):
            # Add virtual node features to all real nodes
            if self.use_virtual_node:
                x = x + vn_embed[batch]

            # GATv2 convolution with edge features
            x_new = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x_new = self.bns[i](x_new)
            x_new = F.gelu(x_new)

            # Residual connection (after first layer)
            if i > 0:
                x = x + self.dropout(x_new)
            else:
                x = self.dropout(x_new)

            # Update virtual node from all real nodes
            if self.use_virtual_node and i < self.num_layers - 1:
                vn_update = global_mean_pool(x, batch)
                vn_embed = vn_embed + self.virtual_node_mlp(vn_update)

        # Multi-scale readout (Mean + Max + Sum)
        pool_mean = global_mean_pool(x, batch)
        pool_max = global_max_pool(x, batch)
        pool_sum = global_add_pool(x, batch)

        pooled = torch.cat([pool_mean, pool_max, pool_sum], dim=1)
        return self.readout_proj(pooled)


class GINEMolecularEncoder(nn.Module):
    """
    GINEConv encoder (kept as a mutation option for architecture search).
    """
    def __init__(self, node_dim=9, edge_dim=3, hidden_dim=128, num_layers=2,
                 dropout=0.2, use_virtual_node=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.use_virtual_node = use_virtual_node
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_virtual_node:
            self.virtual_node_embedding = nn.Embedding(1, hidden_dim)
            self.virtual_node_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.readout_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.node_encoder(x.float())
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr.float())

        if self.use_virtual_node:
            num_graphs = batch.max().item() + 1
            vn_embed = self.virtual_node_embedding(
                torch.zeros(num_graphs, dtype=torch.long, device=x.device)
            )

        for i in range(self.num_layers):
            if self.use_virtual_node:
                x = x + vn_embed[batch]
            x_new = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x_new = self.bns[i](x_new)
            x_new = F.gelu(x_new)
            if i > 0:
                x = x + self.dropout(x_new)
            else:
                x = self.dropout(x_new)
            if self.use_virtual_node and i < self.num_layers - 1:
                vn_update = global_mean_pool(x, batch)
                vn_embed = vn_embed + self.virtual_node_mlp(vn_update)

        pool_mean = global_mean_pool(x, batch)
        pool_max = global_max_pool(x, batch)
        pool_sum = global_add_pool(x, batch)
        pooled = torch.cat([pool_mean, pool_max, pool_sum], dim=1)
        return self.readout_proj(pooled)


# ============================================================================
# Component 2: Tabular Feature Encoder
# ============================================================================

class TabularEncoder(nn.Module):
    """
    MLP encoder for tabular membrane/operating descriptors.
    """
    def __init__(self, input_dim=24, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_d = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Component 3: Cross-Modal Attention Fusion
# ============================================================================

class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between tabular and graph embeddings.
    Dynamically weighs modalities per sample using a learned gate.
    
    This is the KEY NOVELTY — replaces the simple α × table + (1-α) × graph.
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Table attends to Graph
        self.cross_attn_t2g = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_t2g = nn.LayerNorm(d_model)

        # Graph attends to Table
        self.cross_attn_g2t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_g2t = nn.LayerNorm(d_model)

        # Gating network — produces per-sample fusion weight
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, table_feat, graph_feat):
        # Reshape for attention: [batch, 1, d_model]
        t = table_feat.unsqueeze(1)
        g = graph_feat.unsqueeze(1)

        # Table attends to Graph (what molecular features matter for this membrane config?)
        t_enriched, _ = self.cross_attn_t2g(t, g, g)
        t_enriched = self.norm_t2g(t + t_enriched).squeeze(1)

        # Graph attends to Table (what operating conditions matter for this molecule?)
        g_enriched, _ = self.cross_attn_g2t(g, t, t)
        g_enriched = self.norm_g2t(g + g_enriched).squeeze(1)

        # Learned gate: per-sample dynamic weighting
        gate_input = torch.cat([t_enriched, g_enriched], dim=-1)
        gate_weight = self.gate(gate_input)  # [batch, 1]

        # Gated fusion
        fused = gate_weight * t_enriched + (1 - gate_weight) * g_enriched
        return self.proj(fused)


class ConcatFusion(nn.Module):
    """Simple concatenation fusion (baseline for ablation)."""
    def __init__(self, d_model=128, dropout=0.1, **kwargs):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, table_feat, graph_feat):
        return self.proj(torch.cat([table_feat, graph_feat], dim=-1))


class GatedFusion(nn.Module):
    """Gated fusion without cross-attention (simpler alternative)."""
    def __init__(self, d_model=128, dropout=0.1, **kwargs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, table_feat, graph_feat):
        gate_input = torch.cat([table_feat, graph_feat], dim=-1)
        g = self.gate(gate_input)
        fused = g * table_feat + (1 - g) * graph_feat
        return self.proj(fused)


# ============================================================================
# Component 4: Prediction Head
# ============================================================================

class PredictionHead(nn.Module):
    """
    Final regression head with optional MC-Dropout for uncertainty.
    """
    def __init__(self, input_dim=128, hidden_dim=64, dropout=0.2, init_bias=72.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        if init_bias is not None:
            nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# Component 5: Physics-Constrained Loss
# ============================================================================

class PhysicsConstrainedLoss(nn.Module):
    """
    Composite loss = MSE + physics penalty terms.
    Encodes domain knowledge about membrane separation directly into training.
    """
    def __init__(self, lambda_steric=0.1, lambda_bounds=0.01,
                 use_huber=False, huber_delta=5.0):
        super().__init__()
        self.lambda_steric = lambda_steric
        self.lambda_bounds = lambda_bounds
        self.use_huber = use_huber
        if use_huber:
            self.base_loss = nn.HuberLoss(delta=huber_delta)
        else:
            self.base_loss = nn.MSELoss()

    def forward(self, pred, target, steric_ratios=None):
        # Base data loss
        loss = self.base_loss(pred, target)

        # Physics constraint 1: Bounds [0, 100]
        if self.lambda_bounds > 0:
            bounds_penalty = (
                torch.mean(F.relu(-pred) ** 2) +
                torch.mean(F.relu(pred - 100.0) ** 2)
            )
            loss = loss + self.lambda_bounds * bounds_penalty

        # Physics constraint 2: Steric exclusion
        # If solute_radius / pore_radius >= 1.0, rejection should be high (> 85%)
        if self.lambda_steric > 0 and steric_ratios is not None:
            steric_mask = steric_ratios >= 1.0
            if steric_mask.any():
                steric_penalty = torch.mean(F.relu(85.0 - pred[steric_mask]) ** 2)
                loss = loss + self.lambda_steric * steric_penalty

        return loss


# ============================================================================
# THE COMPLETE MODEL: PhysiChemNet
# ============================================================================

class PhysiChemNet(nn.Module):
    """
    PhysiChemNet: Complete end-to-end multimodal model for membrane rejection prediction.
    
    REPLACES the entire DynamicNet gradient boosting framework.
    Single forward pass, single loss, single optimizer.
    
    Architecture:
        Tabular Features → TabularEncoder → 128-D
        Molecular Graph  → GATv2Encoder   → 128-D  
                                    ↓
                        CrossModalAttentionFusion → 128-D
                                    ↓
                            PredictionHead → scalar (rejection %)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Graph encoder
        gnn_type = config.get('gnn_type', 'gatv2')
        if gnn_type == 'gatv2':
            self.graph_encoder = GATv2MolecularEncoder(
                node_dim=config.get('node_dim', 9),
                edge_dim=config.get('edge_dim', 3),
                hidden_dim=config.get('hidden_dim', 128),
                num_layers=config.get('gnn_layers', 2),
                heads=config.get('gnn_heads', 4),
                dropout=config.get('gnn_dropout', 0.2),
                use_virtual_node=config.get('use_virtual_node', True)
            )
        else:  # gine
            self.graph_encoder = GINEMolecularEncoder(
                node_dim=config.get('node_dim', 9),
                edge_dim=config.get('edge_dim', 3),
                hidden_dim=config.get('hidden_dim', 128),
                num_layers=config.get('gnn_layers', 2),
                dropout=config.get('gnn_dropout', 0.2),
                use_virtual_node=config.get('use_virtual_node', True)
            )

        # Tabular encoder
        self.table_encoder = TabularEncoder(
            input_dim=config.get('table_dim', 24),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('table_layers', 2),
            dropout=config.get('table_dropout', 0.2)
        )

        # Fusion
        fusion_type = config.get('fusion_type', 'cross_attention')
        hidden_dim = config.get('hidden_dim', 128)
        fusion_dropout = config.get('fusion_dropout', 0.1)
        if fusion_type == 'cross_attention':
            self.fusion = CrossModalAttentionFusion(
                d_model=hidden_dim,
                nhead=config.get('fusion_heads', 4),
                dropout=fusion_dropout
            )
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(d_model=hidden_dim, dropout=fusion_dropout)
        else:
            self.fusion = ConcatFusion(d_model=hidden_dim, dropout=fusion_dropout)

        # Prediction head
        self.pred_head = PredictionHead(
            input_dim=hidden_dim,
            hidden_dim=config.get('pred_hidden', 64),
            dropout=config.get('pred_dropout', 0.2)
        )

    def forward(self, table_data, graph_data):
        table_embed = self.table_encoder(table_data)
        graph_embed = self.graph_encoder(graph_data)
        fused = self.fusion(table_embed, graph_embed)
        pred = self.pred_head(fused)
        return pred

    def predict_with_uncertainty(self, table_data, graph_data, n_samples=30):
        """MC-Dropout uncertainty estimation."""
        self.train()  # Keep dropout ON
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(table_data, graph_data)
                predictions.append(pred)
        self.eval()
        preds = torch.stack(predictions, dim=0)
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
        return mean_pred, std_pred

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
