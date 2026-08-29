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
    torch.backends.cudnn.deterministic = True  # CUDA RESULTS
    torch.backends.cudnn.benchmark = False     # Note: processed parameter

set_seed(42)  # Set global random seed
# Note: processed parameter
file_path = "../../data/processed/MemTrOC-Dataset.csv"
data = pd.read_csv(file_path)

# Note: processed parameter
X = data.iloc[:, 4:23].values  # 19
y = data.iloc[:, 23].values  # Note: processed parameter

# #  Value
# scaler_X = MinMaxScaler()
# X_normalized = scaler_X.fit_transform(X)
# # #
# X_train, X_test, y_train, y_test = train_test_split(
#     X_normalized, y, test_size=0.2, random_state=42
# )
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.1, random_state=42
# )


# Note: processed parameter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Train Set
scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)  # Train Set scaler
X_test = scaler_X.transform(X_test)  # Train Set scaler

# Convert to PyTorch Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# TensorDataset
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# Create DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Note: processed parameter
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


# 、
input_dim = X.shape[1]
model = ComplexNN(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# CPU GPU
device = torch.device("cpu")  # CPU
model.to(device)


# Note: processed parameter
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


# Metric
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


# Note: processed parameter
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

    # Note: processed parameter
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

# Note: processed parameter
test_metrics = evaluate(model, test_loader)

# RESULTS
print("Test Set Metrics:")
print(f"R2: {test_metrics['R2']:.4f}")
print(f"MAE: {test_metrics['MAE']:.4f}")
print(f"MSE: {test_metrics['MSE']:.4f}")
print(f"RMSE: {test_metrics['RMSE']:.4f}")
print(f"MAPE: {test_metrics['MAPE']:.4f}%")