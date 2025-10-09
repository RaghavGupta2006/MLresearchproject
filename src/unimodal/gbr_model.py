import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor  # 使用 GBR
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
file_path = "../../data/processed/MemTrOC-Dataset.csv"  # 假设文件为 CSV 格式
data = pd.read_csv(file_path)

# 第5列到第23列为特征（索引4到22），第24列为标签（索引23）
X = data.iloc[:, 4:23]  # 第5列到第23列（索引4到22）
y = data.iloc[:, 23]    # 第24列（索引23）

# 2. 划分数据集
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2 / 0.9, random_state=42)

# 3. 数据标准化
scaler = StandardScaler()  # 创建标准化对象
scaler.fit(X_train)  # 仅使用训练数据拟合标准化参数

# 应用标准化到所有数据集
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],  # 树的数量
    'max_depth': [3, 5, 7],          # 树的最大深度
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'subsample': [0.7, 0.8, 0.9],    # 训练实例的比例
}
# Best parameters found:  {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 6,
# 'n_estimators': 300, 'subsample': 0.8} 5. 初始化模型
model = GradientBoostingRegressor(random_state=41)


# # 6. 使用 GridSearchCV 进行网格搜索和交叉验证
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     cv=5,  # 5折交叉验证
#     n_jobs=-1,  # 并行计算
#     verbose=1  # 输出详细进度信息
# )

grid_search = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, n_estimators=300, alpha=0.7, random_state=41)


# 7. 训练模型
grid_search.fit(X_train_scaled, y_train)

# 8. 输出最优参数
# print("Best parameters found: ", grid_search.best_params_)
# print("Best RMSE score: ", -grid_search.best_score_)

# 9. 使用最优参数训练模型
# best_model = grid_search.best_estimator_
best_model = grid_search
# 10. 模型预测
y_train_pred = best_model.predict(X_train_scaled)
y_val_pred = best_model.predict(X_val_scaled)
y_test_pred = best_model.predict(X_test_scaled)


# 11. 定义计算指标的函数

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    Note: It assumes that y_true does not contain zeros to avoid division by zero.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0  # Avoid division by zero
    y_true = y_true[non_zero_indices]
    y_pred = y_pred[non_zero_indices]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}

# 12. 输出结果
results = {
    "Training Set": calculate_metrics(y_train, y_train_pred),
    "Validation Set": calculate_metrics(y_val, y_val_pred),
    "Test Set": calculate_metrics(y_test, y_test_pred)
}

# 打印结果
for dataset, metrics in results.items():
    print(f"{dataset} Metrics:")
    print(f"R2: {metrics['R2']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.4f}")
    print("\n")

# Training Set Metrics:
# R2: 0.9952
# MAE: 1.3239
# MSE: 3.4474
# RMSE: 1.8567
# MAPE: 2.8834
#
#
# Validation Set Metrics:
# R2: 0.8454
# MAE: 6.9334
# MSE: 132.6417
# RMSE: 11.5170
# MAPE: 21.7236
#
#
# Test Set Metrics:
# R2: 0.8077
# MAE: 7.6759
# MSE: 149.7084
# RMSE: 12.2355
# MAPE: 30.9075