import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import TableGraphDataset
from src.utils.smiles2graph import create_graph_data_from_smiles
from src.utils.physics_features import extract_features_and_labels
from torch_geometric.data import Batch
import argparse
import time
from models.weaklearner import MLP_GNN
from models.ensemblemodel import DynamicNetForMLPGNN
from torch.optim import SGD, Adam

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

"""
Multimodal Training Pipeline: Tabular Descriptors + Molecular Graph Data
"""
parser = argparse.ArgumentParser(description="Multimodal Training Pipeline for MolGBN-OPR")

# Model hyper-parameters
parser.add_argument('--feat_d', type=int, help='Feature dimension', default=24)
parser.add_argument('--hidden_d', type=int, help='Hidden layer dimension', default=128)
parser.add_argument('--table_dim_in', type=int, default=24, help='Tabular input dimension')
parser.add_argument('--table_dim_hidden', type=int, default=128, help='Hidden dimension for tabular MLP')
parser.add_argument('--gnn_input_dim', type=int, default=9, help='Node feature dimension')
parser.add_argument('--edge_dim', type=int, default=3, help='Edge feature dimension for GINE')
parser.add_argument('--gnn_type', type=str, default='gine', choices=['gine', 'gcn'], help='GNN backbone type')
parser.add_argument('--use_physics', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable physics-informed domain features')
parser.add_argument('--out_dim', type=int, default=128, help='Output dimension for feature extractors')
parser.add_argument('--gnn_hidden', type=int, default=128, help='Hidden dimension for GNN')
parser.add_argument('--combined_dim', type=int, default=128, help='Combined feature dimension')
parser.add_argument('--dim_hidden1', type=int, default=128, help='First hidden dimension after fusion')
parser.add_argument('--dim_hidden2', type=int, default=128, help='Second hidden dimension after fusion')

# Optimization & Regularization
parser.add_argument('--boost_rate', type=float, help='Boosting rate', default=1.0)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--L2', type=float, help='L2 regularization coefficient', default=1.0e-2)

# Training parameters
parser.add_argument('--num_nets', type=int, help='Number of weak learners (stages)', default=8)
parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
parser.add_argument('--epochs_per_stage', type=int, help='Epochs per stage', default=100)
parser.add_argument('--correct_epoch', type=int, help='Epochs for corrective step', default=100)

# Paths and flags
parser.add_argument('--data', type=str, help='Path to data')
parser.add_argument('--tr', type=str, help='Path to training data')
parser.add_argument('--te', type=str, help='Path to testing data')
parser.add_argument('--out_f', type=str, help='Output checkpoint path', default='checkpoint/best_GrowTableGINE_enhanced.pth')

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
    with torch.no_grad():
        for x, graph_data, y in loader:
            if args.cuda:
                x = x.to(device)
                graph_data = graph_data.to(device)
            _, out = net_ensemble.forward(x, graph_data)
            y_np = y.cpu().numpy().reshape(len(y), 1)
            out_np = out.cpu().numpy().reshape(len(y), 1)
            loss += mean_squared_error(y_np, out_np) * len(y)
            total += len(y)
    return np.sqrt(loss / total)


def my_collate(batch):
    table_features, graph_data, labels = zip(*batch)
    table_features = torch.stack(table_features, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    graph_data = Batch.from_data_list(graph_data)
    return table_features, graph_data, labels


def get_predictions(net_ensemble, loader):
    net_ensemble.to_eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, graph_data, y in loader:
            if args.cuda:
                x = x.to(device)
                graph_data = graph_data.to(device)
                y = y.to(device)
            _, preds = net_ensemble.forward(x, graph_data)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def worker_init_fn(worker_id):
    np.random.seed(41 + worker_id)
    random.seed(41 + worker_id)


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
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "processed", "MemTrOC-Dataset.csv")
    data = pd.read_csv(file_path)

    if args.out_f.startswith('../'):
        args.out_f = os.path.join(base_dir, args.out_f[3:])
    elif not os.path.isabs(args.out_f):
        args.out_f = os.path.join(base_dir, args.out_f)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_f)), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    # Extract features and labels
    X, y, smiles_list = extract_features_and_labels(data, use_physics=args.use_physics)
    args.table_dim_in = X.shape[1]
    args.feat_d = X.shape[1]
    print(f"Tabular features dimension: {X.shape[1]} (Physics-Informed: {args.use_physics})")

    # Train / Val / Test Split
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X, y, smiles_list, test_size=0.1, random_state=41
    )
    X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(
        X_train, y_train, smiles_train, test_size=0.2 / 0.9, random_state=41
    )

    # Normalize on training split
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

    train_dataset = TableGraphDataset(X_train_t, smiles_train, y_train_t, create_graph_data_from_smiles)
    val_dataset = TableGraphDataset(X_val_t, smiles_val, y_val_t, create_graph_data_from_smiles)
    test_dataset = TableGraphDataset(X_test_t, smiles_test, y_test_t, create_graph_data_from_smiles)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=my_collate, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=my_collate, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=my_collate, worker_init_fn=worker_init_fn)

    N = len(X_train)
    best_rmse = 1e6
    val_rmse = best_rmse
    best_stage = args.num_nets - 1
    c0 = float(y_train.mean())
    net_ensemble = DynamicNetForMLPGNN(c0, args.boost_rate)
    loss_f1 = nn.MSELoss()
    loss_models = torch.zeros((args.num_nets, 3))

    for stage in range(args.num_nets):
        t0 = time.time()
        model = MLP_GNN.get_model(stage, args)
        if args.cuda:
            model = model.to(device)

        optimizer = get_optim(model.parameters(), args.lr, args.L2)
        net_ensemble.to_train()
        stage_mdlloss = []

        for epoch in range(args.epochs_per_stage):
            for i, (x, graph_data, y) in enumerate(train_loader):
                if args.cuda:
                    x = x.to(device)
                    graph_data = graph_data.to(device)
                    y = y.to(device).view(-1, 1)
                else:
                    y = y.view(-1, 1)

                middle_feat, out = net_ensemble.forward(x, graph_data)
                out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
                if args.cuda:
                    out = out.to(device)

                grad_direction = -(out - y)
                _, out_sub = model(x, graph_data, middle_feat)
                out_sub = out_sub.view(-1, 1)
                loss = loss_f1(net_ensemble.boost_rate * out_sub, grad_direction)

                model.zero_grad()
                loss.backward()
                optimizer.step()
                stage_mdlloss.append(loss.item() * len(y))

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
                for i, (x, graph_data, y) in enumerate(train_loader):
                    if args.cuda:
                        x = x.to(device)
                        graph_data = graph_data.to(device)
                        y = y.to(device).view(-1, 1)
                    else:
                        y = y.view(-1, 1)

                    _, out_grad = net_ensemble.forward_grad(x, graph_data)
                    out_grad = torch.as_tensor(out_grad, dtype=torch.float32).view(-1, 1)
                    loss = loss_f1(out_grad, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    stage_loss.append(loss.item() * len(y))

        elapsed_tr = time.time() - t0
        sl = 0
        if stage_loss:
            sl = np.sqrt(np.sum(stage_loss) / N)

        print(f'Stage {stage} ({elapsed_tr:.1f}s) | Model RMSE: {sml:.4f}, Corrective RMSE: {sl:.4f}')

        net_ensemble.to_file(args.out_f)
        if args.cuda:
            net_ensemble.to_cuda()
        net_ensemble = DynamicNetForMLPGNN.from_file(args.out_f, lambda s: MLP_GNN.get_model(s, args))
        if args.cuda:
            net_ensemble.to_cuda()
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
    print(f'Best Validation Stage: {best_stage} (RMSE@Train: {tr_rmse:.4f}, RMSE@Test: {te_rmse:.4f})')

    # Load best validation stage model
    net_ensemble = DynamicNetForMLPGNN.from_file(args.out_f, lambda s: MLP_GNN.get_model(s, args))
    if args.cuda:
        net_ensemble.to_cuda()

    train_pred, train_true = get_predictions(net_ensemble, train_loader)
    val_pred, val_true = get_predictions(net_ensemble, val_loader)
    test_pred, test_true = get_predictions(net_ensemble, test_loader)

    R2_train = r2_score(train_true, train_pred)
    R2_val = r2_score(val_true, val_pred)
    R2_test = r2_score(test_true, test_pred)

    rmse_train = np.sqrt(mean_squared_error(train_true, train_pred))
    rmse_val = np.sqrt(mean_squared_error(val_true, val_pred))
    rmse_test = np.sqrt(mean_squared_error(test_true, test_pred))

    mae_train = mean_absolute_error(train_true, train_pred)
    mae_val = mean_absolute_error(val_true, val_pred)
    mae_test = mean_absolute_error(test_true, test_pred)

    print("\n------------------------ FINAL RESULTS ------------------------")
    print(f'Train -> R2: {R2_train:.4f} | RMSE: {rmse_train:.4f}% | MAE: {mae_train:.4f}%')
    print(f'Val   -> R2: {R2_val:.4f} | RMSE: {rmse_val:.4f}% | MAE: {mae_val:.4f}%')
    print(f'Test  -> R2: {R2_test:.4f} | RMSE: {rmse_test:.4f}% | MAE: {mae_test:.4f}%\n')

    # Save metrics to log file
    log_path = os.path.join(base_dir, 'checkpoint', 'log.txt')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("{ Multimodal GINE Training Results }\n")
        f.write("=" * 60 + "\n\n")
        f.write("| Metric | Train Set | Validation Set | Test Set |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| R2 Score | {R2_train:.4f} | {R2_val:.4f} | {R2_test:.4f} |\n")
        f.write(f"| RMSE (%) | {rmse_train:.4f} | {rmse_val:.4f} | {rmse_test:.4f} |\n")
        f.write(f"| MAE (%)  | {mae_train:.4f} | {mae_val:.4f} | {mae_test:.4f} |\n")
        f.write("-" * 60 + "\n")
        f.write(f"Best Stage: {best_stage}\n")
        f.write("=" * 60 + "\n\n")
