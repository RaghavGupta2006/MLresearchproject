from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        f1 = 32
        f2 = 32
        c1 = 20
        c2 = 20
        dropout_rate = 0.10
        t1 = 1
        self.t1 = t1
        self.conv1 = nn.Conv1d(1, c1, 4)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(c1, c2, 4)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1 * c2, f1)
        self.fc2 = nn.Linear(f1, f2)
        self.fc3 = nn.Linear(f2, f2)
        self.fc4 = nn.Linear(f2, 1)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = x.reshape(-1, 1, 16)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        temp = self.relu(x)
        x = self.relu(self.fc3(temp))
        if self.t1 == 1:
            x = self.relu(self.fc4(x + temp))
        else:
            x = self.relu(self.fc4(x))

        return x


def get_model():
    net = model()
    return net


class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        f1 = 64
        f2 = 32
        c1 = 32
        c2 = 16
        dropout_rate = 0.2

        self.conv1 = nn.Conv1d(1, c1, 4)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(c1, c2, 4)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(c2 * 4, f1)
        self.fc2 = nn.Linear(f1, f2)
        self.fc3 = nn.Linear(f2, 1)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = x.reshape(-1, 1, 16)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


def get_new_model():
    net = NewModel()
    return net


# Enhanced Regression DNN Model
class RegressionDNN(nn.Module):
    def __init__(self, in_features=19):
        super(RegressionDNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x


# GrowNet Architecture
class GrowNet(nn.Module):
    def __init__(self, num_trees=7, in_features=19):
        super().__init__()
        self.trees = nn.ModuleList([RegressionDNN(in_features=in_features) for _ in range(num_trees)])
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_trees)])

    def partial_predict(self, x, num_trees):
        with torch.no_grad():
            pred = torch.zeros(x.size(0), 1).to(x.device)
            for tree, weight in zip(self.trees[:num_trees], self.weights[:num_trees]):
                pred += weight * tree(x)
        return pred

    def forward(self, x):
        total = torch.zeros(x.size(0), 1).to(x.device)
        for tree, weight in zip(self.trees, self.weights):
            total += weight * tree(x)
        return total


def get_1dcnn_model(num_trees, in_features=19):
    net = GrowNet(num_trees, in_features=in_features)
    return net
