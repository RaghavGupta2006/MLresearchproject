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

        self.conv1 = nn.Conv1d(1, c1, 4)  # Add padding to maintain spatial dimensions
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(c1, c2, 4)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(c2 * 4, f1)  # Adjust input size based on pooling and padding
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


# 定义一个简单的神经网络树
class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # 第二层卷积层
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        # 全连接层
        self.fc = nn.Linear(in_features=32, out_features=1)  # 假设我们只有一个输出值

    def forward(self, x):
        x = x.reshape(-1, 1, 16)
        # 第一层卷积和池化
        x = self.pool1(torch.relu(self.conv1(x)))
        # 第二层卷积和池化
        x = self.pool2(torch.relu(self.conv2(x)))
        # 扁平化处理
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.fc(x)  # 不使用激活函数，因为这是回归任务
        return x


# 定义GrowNet
class GrowNet(nn.Module):
    def __init__(self, num_trees=7):
        super(GrowNet, self).__init__()
        self.trees = nn.ModuleList([RegressionCNN() for _ in range(num_trees)])

    def forward(self, x):
        predictions = []
        # residual = x.clone()
        residual = x
        for tree in self.trees:
            prediction = tree(residual)
            predictions.append(prediction)
            residual = residual - prediction
        return sum(predictions)


def get_1dcnn_model(num_trees):
    net = GrowNet(num_trees)
    return net
