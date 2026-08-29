#!/usr/bin/env python
import argparse
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.weaklearner import MLP_2HL, MLP_3HL
from models.ensemblemodel import DynamicNet
from torch.optim import SGD, Adam

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import random

"""
Training Pipeline for Pure Tabular Baseline (GrowNN)
"""
parser = argparse.ArgumentParser(description="GrowNN Tabular Baseline Training")

# Integer parameters
parser.add_argument('--feat_d', type=int, help='Feature dimension', default=19)
parser.add_argument('--hidden_d', type=int, help='Hidden layer dimension', default=128)

# Optimization parameters
parser.add_argument('--boost_rate', type=float, help='Boosting rate', default=1.0)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--L2', type=float, help='L2 regularization coefficient', default=1.0e-2)

parser.add_argument('--num_nets', type=int, help='Number of networks (stages)', default=5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
parser.add_argument('--epochs_per_stage', type=int, help='Epochs per stage', default=100)
parser.add_argument('--correct_epoch', type=int, help='Epochs for corrective step', default=100)

parser.add_argument('--data', type=str, help='Path to data')
parser.add_argument('--tr', type=str, help='Path to training data')
parser.add_argument('--te', type=str, help='Path to testing data')
parser.add_argument('--out_f', type=str, help='Output file path', default='checkpoint/best_GrowNN.pth')

parser.add_argument('--sparse', action='store_true', help='Use sparse representation')
parser.add_argument('--normalization', type=lambda x: (str(x).lower() == 'true'), default=False, help='Enable normalization')
parser.add_argument('--cv', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable validation check')
parser.add_argument('--cuda', action='store_true', help='Use CUDA acceleration', default=False)

args = parser.parse_args()

if not args.cuda:
    torch.set_num_threads(4)


def get_optim(params, lr, weight_decay):
    return Adam(params, lr=lr, weight_decay=weight_decay)


def root_mse(net_ensemble, loader):
    loss = 0.0
    total = 0
    for x, y in loader:
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        y_np = y.cpu().numpy().reshape(len(y), 1)
        out_np = out.cpu().numpy().reshape(len(y), 1)
        loss += mean_squared_error(y_np, out_np) * len(y)
        total += len(y)
    return np.sqrt(loss / total)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(41)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "processed", "MemTrOC-Dataset.csv")
    data = pd.read_csv(file_path)

    if args.out_f.startswith('../'):
        args.out_f = os.path.join(base_dir, args.out_f[3:])
    elif not os.path.isabs(args.out_f):
        args.out_f = os.path.join(base_dir, args.out_f)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_f)), exist_ok=True)

    X = data.iloc[:, 4:23].values
    y = data.iloc[:, 23].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=41)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2 / 0.9, random_state=41)

    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    N = len(X_train)
    best_rmse = 1e6
    val_rmse = best_rmse
    best_stage = args.num_nets - 1
    c0 = float(y_train.mean())
    net_ensemble = DynamicNet(c0, args.boost_rate)
    loss_f1 = nn.MSELoss()
    loss_models = torch.zeros((args.num_nets, 3))

    for stage in range(args.num_nets):
        t0 = time.time()
        model = MLP_2HL.get_model(stage, args)

        optimizer = get_optim(model.parameters(), args.lr, args.L2)
        net_ensemble.to_train()
        stage_mdlloss = []

        for epoch in range(args.epochs_per_stage):
            for i, (x, y_batch) in enumerate(train_loader):
                middle_feat, out = net_ensemble.forward(x)
                out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
                grad_direction = -(out - y_batch)

                _, out_sub = model(x, middle_feat)
                out_sub = torch.as_tensor(out_sub, dtype=torch.float32).view(-1, 1)
                loss = loss_f1(net_ensemble.boost_rate * out_sub, grad_direction)

                model.zero_grad()
                loss.backward()
                optimizer.step()
                stage_mdlloss.append(loss.item() * len(y_batch))

        net_ensemble.add(model)
        sml = np.sqrt(np.sum(stage_mdlloss) / N)

        # Fully-corrective step
        lr_scaler = 3
        stage_loss = []
        if stage > 0:
            if stage % 15 == 0:
                args.lr /= 2
                args.L2 /= 2
            optimizer = get_optim(net_ensemble.parameters(), args.lr / lr_scaler, args.L2)
            for _ in range(args.correct_epoch):
                stage_loss = []
                for i, (x, y_batch) in enumerate(train_loader):
                    _, out_all = net_ensemble.forward_grad(x)
                    out_all = torch.as_tensor(out_all, dtype=torch.float32).view(-1, 1)
                    loss = loss_f1(out_all, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    stage_loss.append(loss.item() * len(y_batch))

        elapsed_tr = time.time() - t0
        sl = 0
        if stage_loss:
            sl = np.sqrt(np.sum(stage_loss) / N)

        print(f'Stage {stage} ({elapsed_tr:.1f}s) | Model RMSE: {sml:.4f}, Ensemble RMSE: {sl:.4f}')

        net_ensemble.to_file(args.out_f)
        net_ensemble = DynamicNet.from_file(args.out_f, lambda s: MLP_2HL.get_model(s, args))
        net_ensemble.to_eval()

        tr_rmse = root_mse(net_ensemble, train_loader)
        if args.cv:
            val_rmse = root_mse(net_ensemble, val_loader)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_stage = stage

        te_rmse = root_mse(net_ensemble, test_loader)
        print(f'Stage: {stage} -> RMSE@Train: {tr_rmse:.4f}, RMSE@Val: {val_rmse:.4f}, RMSE@Test: {te_rmse:.4f}')
        loss_models[stage, 0], loss_models[stage, 1] = tr_rmse, te_rmse

    tr_rmse, te_rmse = loss_models[best_stage, 0], loss_models[best_stage, 1]
    print(f'Best validation stage: {best_stage} (RMSE@Train: {tr_rmse:.4f}, RMSE@Test: {te_rmse:.4f})')

    net_ensemble = DynamicNet.from_file(args.out_f, lambda b: MLP_2HL.get_model(b, args))

    _, prediction_train = net_ensemble.forward(X_train_t)
    _, prediction_val = net_ensemble.forward(X_val_t)
    _, prediction_test = net_ensemble.forward(X_test_t)

    pred_tr = prediction_train.detach().cpu().numpy()
    pred_va = prediction_val.detach().cpu().numpy()
    pred_te = prediction_test.detach().cpu().numpy()

    R2_train = r2_score(y_train, pred_tr)
    R2_val = r2_score(y_val, pred_va)
    R2_test = r2_score(y_test, pred_te)

    rmse_train = np.sqrt(mean_squared_error(y_train, pred_tr))
    rmse_val = np.sqrt(mean_squared_error(y_val, pred_va))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_te))

    mae_train = mean_absolute_error(y_train, pred_tr)
    mae_val = mean_absolute_error(y_val, pred_va)
    mae_test = mean_absolute_error(y_test, pred_te)

    print("\n------------------------ RESULTS ------------------------")
    print(f'Train -> R2: {R2_train:.4f} | RMSE: {rmse_train:.4f}% | MAE: {mae_train:.4f}%')
    print(f'Val   -> R2: {R2_val:.4f} | RMSE: {rmse_val:.4f}% | MAE: {mae_val:.4f}%')
    print(f'Test  -> R2: {R2_test:.4f} | RMSE: {rmse_test:.4f}% | MAE: {mae_test:.4f}%\n')

    # Save to log file
    log_path = os.path.join(base_dir, 'checkpoint', 'GrowNN_log.txt')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("{ GrowNN Tabular Baseline Results }\n")
        f.write("=" * 60 + "\n\n")
        f.write("| Metric | Train Set | Validation Set | Test Set |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| R2 Score | {R2_train:.4f} | {R2_val:.4f} | {R2_test:.4f} |\n")
        f.write(f"| RMSE (%) | {rmse_train:.4f} | {rmse_val:.4f} | {rmse_test:.4f} |\n")
        f.write(f"| MAE (%)  | {mae_train:.4f} | {mae_val:.4f} | {mae_test:.4f} |\n")
        f.write("-" * 60 + "\n\n")
