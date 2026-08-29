"""
 XGB  '../../data/processed/MemTrOC-Dataset.xlsx'  
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1.
file_path = "../../data/processed/MemTrOC-Dataset.csv"  # CSV
data = pd.read_csv(file_path)

# 5 23 4 22 24 23
X = data.iloc[:, 4:23]  # 5 23 4 22
y = data.iloc[:, 23]  # 24 23

# 2.
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2 / 0.9, random_state=42)

# 3.
scaler = StandardScaler()  # Note: processed parameter
scaler.fit(X_train)  # Note: processed parameter

# Note: processed parameter
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4.
param_grid = {
    'n_estimators': [100, 200, 300],  # Note: processed parameter
    'max_depth': [3, 5, 7],  # Note: processed parameter
    'learning_rate': [0.01, 0.1, 0.2],  # Note: processed parameter
    # 'min_child_weight': [1, 3, 5],   #
    'subsample': [0.7, 0.8, 0.9],  # Note: processed parameter
    'colsample_bytree': [0.7, 0.8, 0.9]  # Note: processed parameter
}
# Best parameters found:  {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300,
# 'subsample': 0.8} 5.
model = XGBRegressor(random_state=42)
# Train Set
# R2: 0.9996
# MAE: 0.3310
# MSE: 0.3116
# RMSE: 0.5582

# Validation Set
# R2: 0.8776
# MAE: 6.1307
# MSE: 105.0044
# RMSE: 10.2472
# Test Set
# R2: 0.8288
# MAE: 7.2889
# MSE: 133.2985
# RMSE: 11.5455

# 6.   GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',  # Metric
    cv=5,  # 5
    n_jobs=-1,  # Note: processed parameter
    verbose=1  # Note: processed parameter
)

# 7.
grid_search.fit(X_train_scaled, y_train)

# 8.
print("Best parameters found: ", grid_search.best_params_)
print("Best RMSE score: ", -grid_search.best_score_)

# 9.
best_model = grid_search.best_estimator_

# 10.
y_train_pred = best_model.predict(X_train_scaled)
y_val_pred = best_model.predict(X_val_scaled)
y_test_pred = best_model.predict(X_test_scaled)


# 11.  Metric
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


# 12.  RESULTS
results = {
    "Training Set": calculate_metrics(y_train, y_train_pred),
    "Validation Set": calculate_metrics(y_val, y_val_pred),
    "Test Set": calculate_metrics(y_test, y_test_pred)
}

# RESULTS
for dataset, metrics in results.items():
    print(f"{dataset} Metrics:")
    print(f"R2: {metrics['R2']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.4f}")
    print("\n")
