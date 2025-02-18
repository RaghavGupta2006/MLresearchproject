import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积结果一致
    torch.backends.cudnn.benchmark = False     # 禁用自动优化

set_seed(42)  # 设置全局种子
# 数据路径
file_path = "../../data/processed/MemTrOC-Dataset.csv"
data = pd.read_csv(file_path)

# 提取特征和标签
X = data.iloc[:, 4:23].values  # 特征（19维）
y = data.iloc[:, 23].values  # 标签

# # 归一化特征值
# scaler_X = MinMaxScaler()
# X_normalized = scaler_X.fit_transform(X)
#
# # 数据集划分
# X_train, X_test, y_train, y_test = train_test_split(
#     X_normalized, y, test_size=0.2, random_state=42
# )
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.1, random_state=42
# )


# 先划分数据集再进行归一化（关键修改！）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 只在训练集上拟合归一化器
scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)  # 使用训练集的scaler
X_test = scaler_X.transform(X_test)  # 使用训练集的scaler

# 转换为 PyTorch Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建 TensorDataset
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# 创建 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义模型
class ComplexNN(nn.Module):
    def __init__(self, input_dim):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
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


# 初始化模型、损失函数和优化器
input_dim = X.shape[1]
model = ComplexNN(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 设备选择（CPU或GPU）
device = torch.device("cpu")  # 假设使用CPU
model.to(device)


# 评估函数
def evaluate(model, data_loader):
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.to(device))
            y_pred = outputs.cpu().numpy().flatten()
            y_true = labels.numpy().flatten()
            y_true_list.extend(y_true)
            y_pred_list.extend(y_pred)
    return calculate_metrics(np.array(y_true_list), np.array(y_pred_list))


# 计算指标
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def calculate_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }


# 训练模型
n_epochs = 500
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    batch_losses = []
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)

    # 验证
    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            val_batch_losses.append(loss.item())
    val_loss = np.mean(val_batch_losses)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 测试模型
test_metrics = evaluate(model, test_loader)

# 打印结果
print("Test Set Metrics:")
print(f"R2: {test_metrics['R2']:.4f}")
print(f"MAE: {test_metrics['MAE']:.4f}")
print(f"MSE: {test_metrics['MSE']:.4f}")
print(f"RMSE: {test_metrics['RMSE']:.4f}")
print(f"MAPE: {test_metrics['MAPE']:.4f}%")