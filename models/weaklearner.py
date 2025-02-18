import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_2HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, bn=True):
        super(MLP_2HL, self).__init__()
        self.in_layer = nn.Linear(dim_in, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.0)
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
        # print(x.shape)
        out = self.lrelu(self.in_layer(x))
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
