import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .splinear import SpLinear

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