import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttnFusion(nn.Module):
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


# 使用示例
if __name__ == "__main__":
    model = SelfAttnFusion(d_model=128, nhead=8)
    table_feat = torch.randn(64, 128)
    graph_feat = torch.randn(64, 128)
    output = model(table_feat, graph_feat)
    print(output.shape)  # 输出: torch.Size([64, 128])