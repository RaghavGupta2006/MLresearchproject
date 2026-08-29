import sys
import os
import argparse
import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import TableGraphDataset
from src.utils.smiles2graph import create_graph_data_from_smiles
from src.utils.physics_features import extract_features_and_labels
from models.weaklearner import MLP_GNN
from models.ensemblemodel import DynamicNetForMLPGNN

parser = argparse.ArgumentParser(description="5-Fold Cross-Validation Ensembling for MolGBN-OPR")

# Model hyper-parameters
parser.add_argument('--table_dim_in', type=int, default=24, help='Tabular input dimension')
parser.add_argument('--table_dim_hidden', type=int, default=128, help='Hidden dimension for tabular MLP')
parser.add_argument('--gnn_input_dim', type=int, default=9, help='Node feature dimension')
parser.add_argument('--edge_dim', type=int, default=3, help='Edge feature dimension for GINE')
parser.add_argument('--gnn_type', type=str, default='gine', choices=['gine', 'gcn'], help='GNN backbone type')
parser.add_argument('--use_physics', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable physics-informed features')
parser.add_argument('--out_dim', type=int, default=128, help='Output dimension for feature extractors')
parser.add_argument('--gnn_hidden', type=int, default=128, help='Hidden dimension for GNN')
parser.add_argument('--combined_dim', type=int, default=128, help='Combined feature dimension')
parser.add_argument('--dim_hidden1', type=int, default=128, help='First hidden dimension after fusion')
parser.add_argument('--dim_hidden2', type=int, default=128, help='Second hidden dimension after fusion')

# Optimization & Regularization
parser.add_argument('--boost_rate', type=float, default=1.0, help='Boosting rate')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--L2', type=float, default=0.01, help='L2 weight decay')
parser.add_argument('--num_nets', type=int, default=3, help='Number of weak learners per fold')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epochs_per_stage', type=int, default=60, help='Epochs per stage')
parser.add_argument('--correct_epoch', type=int, default=60, help='Epochs for corrective step')

# K-Fold & Dataset settings
parser.add_argument('--k_fold', type=int, default=5, help='Number of cross-validation folds')
parser.add_argument('--sparse', action='store_true', help='Use sparse representation')
parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA acceleration')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()

if not args.cuda:
    torch.set_num_threads(4)


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


def my_collate(batch):
    table_features, graph_data, labels = zip(*batch)
    table_features = torch.stack(table_features, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    graph_data = Batch.from_data_list(graph_data)
    return table_features, graph_data, labels


def get_optim(params, lr, weight_decay):
    return Adam(params, lr=lr, weight_decay=weight_decay)


def root_mse(net_ensemble, loader, device, use_cuda):
    loss = 0.0
    total = 0
    with torch.no_grad():
        for x, graph_data, y in loader:
            if use_cuda:
                x = x.to(device)
                graph_data = graph_data.to(device)
            _, out = net_ensemble.forward(x, graph_data)
            y_np = y.cpu().numpy().reshape(len(y), 1)
            out_np = out.cpu().numpy().reshape(len(y), 1)
            loss += mean_squared_error(y_np, out_np) * len(y)
            total += len(y)
    return np.sqrt(loss / total)


def get_predictions(net_ensemble, loader, device, use_cuda):
    net_ensemble.to_eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, graph_data, y in loader:
            if use_cuda:
                x = x.to(device)
                graph_data = graph_data.to(device)
            _, preds = net_ensemble.forward(x, graph_data)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy().squeeze()


def train_single_fold(fold_idx, X_tr, y_tr, smiles_tr, X_va, y_va, smiles_va, X_te, y_te, smiles_te, device, args):
    print(f"\n{'='*20} Training Fold {fold_idx + 1} / {args.k_fold} {'='*20}")
    set_seed(args.seed + fold_idx)

    scaler = MinMaxScaler()
    X_tr_norm = scaler.fit_transform(X_tr)
    X_va_norm = scaler.transform(X_va)
    X_te_norm = scaler.transform(X_te)

    train_ds = TableGraphDataset(torch.tensor(X_tr_norm, dtype=torch.float32), smiles_tr, torch.tensor(y_tr, dtype=torch.float32).view(-1, 1), create_graph_data_from_smiles)
    val_ds = TableGraphDataset(torch.tensor(X_va_norm, dtype=torch.float32), smiles_va, torch.tensor(y_va, dtype=torch.float32).view(-1, 1), create_graph_data_from_smiles)
    test_ds = TableGraphDataset(torch.tensor(X_te_norm, dtype=torch.float32), smiles_te, torch.tensor(y_te, dtype=torch.float32).view(-1, 1), create_graph_data_from_smiles)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)

    N = len(X_tr)
    c0 = float(y_tr.mean())
    net_ensemble = DynamicNetForMLPGNN(c0, args.boost_rate)
    loss_fn = nn.MSELoss()

    best_val_rmse = 1e6
    best_stage = 0
    fold_ckpt_path = os.path.join(base_dir, "checkpoint", f"best_GINE_fold{fold_idx}.pth")

    for stage in range(args.num_nets):
        t0 = time.time()
        model = MLP_GNN.get_model(stage, args)
        if args.cuda:
            model = model.to(device)

        optimizer = get_optim(model.parameters(), args.lr, args.L2)
        net_ensemble.to_train()
        stage_loss = []

        # Stage residual fitting
        for epoch in range(args.epochs_per_stage):
            for x, graph_data, y in train_loader:
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
                loss = loss_fn(net_ensemble.boost_rate * out_sub, grad_direction)

                model.zero_grad()
                loss.backward()
                optimizer.step()
                stage_loss.append(loss.item() * len(y))

        net_ensemble.add(model)

        # Corrective step
        if stage > 0:
            lr_decay = 2 if stage % 2 == 0 else 1
            corr_optimizer = get_optim(net_ensemble.parameters(), (args.lr / 3) / lr_decay, args.L2)
            for _ in range(args.correct_epoch):
                for x, graph_data, y in train_loader:
                    if args.cuda:
                        x = x.to(device)
                        graph_data = graph_data.to(device)
                        y = y.to(device).view(-1, 1)
                    else:
                        y = y.view(-1, 1)

                    _, out_all = net_ensemble.forward_grad(x, graph_data)
                    out_all = torch.as_tensor(out_all, dtype=torch.float32).view(-1, 1)
                    loss_corr = loss_fn(out_all, y)

                    corr_optimizer.zero_grad()
                    loss_corr.backward()
                    corr_optimizer.step()

        elapsed = time.time() - t0
        net_ensemble.to_eval()
        tr_rmse = root_mse(net_ensemble, train_loader, device, args.cuda)
        val_rmse = root_mse(net_ensemble, val_loader, device, args.cuda)
        te_rmse = root_mse(net_ensemble, test_loader, device, args.cuda)

        print(f"Fold {fold_idx+1} | Stage {stage} ({elapsed:.1f}s) | Train RMSE: {tr_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Test RMSE: {te_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_stage = stage
            net_ensemble.to_file(fold_ckpt_path)

    print(f"Fold {fold_idx+1} Best Stage: {best_stage} (Best Val RMSE: {best_val_rmse:.4f})")

    # Load best checkpoint for this fold
    best_fold_model = DynamicNetForMLPGNN.from_file(fold_ckpt_path, lambda s: MLP_GNN.get_model(s, args))
    if args.cuda:
        best_fold_model.to_cuda()

    val_preds, val_targets = get_predictions(best_fold_model, val_loader, device, args.cuda)
    test_preds, test_targets = get_predictions(best_fold_model, test_loader, device, args.cuda)

    return val_preds, val_targets, test_preds, test_targets, fold_ckpt_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Executing 5-Fold Cross-Validation on: {device}")

    data_file = os.path.join(base_dir, "data", "processed", "MemTrOC-Dataset.csv")
    df = pd.read_csv(data_file)

    X_all, y_all, smiles_all = extract_features_and_labels(df, use_physics=args.use_physics)
    args.table_dim_in = X_all.shape[1]
    print(f"Tabular features dimension: {X_all.shape[1]} (Physics-Informed: {args.use_physics})")

    # Hold out a fixed 10% test set for overall ensemble benchmark
    X_dev, X_test, y_dev, y_test, smiles_dev, smiles_test = train_test_split(
        X_all, y_all, smiles_all, test_size=0.1, random_state=41
    )

    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)

    oof_predictions = np.zeros(len(X_dev))
    oof_targets = np.zeros(len(X_dev))
    all_test_predictions = []

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_dev)):
        X_tr, y_tr, smiles_tr = X_dev[train_idx], y_dev[train_idx], smiles_dev[train_idx]
        X_va, y_va, smiles_va = X_dev[val_idx], y_dev[val_idx], smiles_dev[val_idx]

        va_preds, va_targs, te_preds, te_targs, ckpt = train_single_fold(
            fold_idx, X_tr, y_tr, smiles_tr, X_va, y_va, smiles_va, X_test, y_test, smiles_test, device, args
        )

        oof_predictions[val_idx] = va_preds
        oof_targets[val_idx] = va_targs
        all_test_predictions.append(te_preds)

        fold_r2 = r2_score(te_targs, te_preds)
        fold_rmse = np.sqrt(mean_squared_error(te_targs, te_preds))
        fold_mae = mean_absolute_error(te_targs, te_preds)
        fold_metrics.append((fold_r2, fold_rmse, fold_mae))
        print(f"Fold {fold_idx+1} Test Set -> R2: {fold_r2:.4f}, RMSE: {fold_rmse:.4f}, MAE: {fold_mae:.4f}")

    # Compute Out-of-Fold (OOF) Overall Metrics
    oof_r2 = r2_score(oof_targets, oof_predictions)
    oof_rmse = np.sqrt(mean_squared_error(oof_targets, oof_predictions))
    oof_mae = mean_absolute_error(oof_targets, oof_predictions)

    # Compute Soft-Ensemble Test Predictions (Average of 5 Folds)
    ensemble_test_pred = np.mean(all_test_predictions, axis=0)
    ens_r2 = r2_score(y_test, ensemble_test_pred)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
    ens_mae = mean_absolute_error(y_test, ensemble_test_pred)

    print("\n" + "="*70)
    print("                5-FOLD CROSS-VALIDATION FINAL RESULTS                ")
    print("="*70)
    print(f"Out-of-Fold (OOF) CV -> R2: {oof_r2:.4f} | RMSE: {oof_rmse:.4f}% | MAE: {oof_mae:.4f}%")
    print("-" * 70)
    for i, (r2_f, rmse_f, mae_f) in enumerate(fold_metrics):
        print(f"  Fold {i+1} Test Set -> R2: {r2_f:.4f} | RMSE: {rmse_f:.4f}% | MAE: {mae_f:.4f}%")
    print("-" * 70)
    print(f"[*] 5-FOLD ENSEMBLE TEST -> R2: {ens_r2:.4f} | RMSE: {ens_rmse:.4f}% | MAE: {ens_mae:.4f}%")
    print("="*70 + "\n")

    # Save to log file
    log_file = os.path.join(base_dir, "checkpoint", "log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*60 + "\n")
        f.write("{ 5-Fold Cross-Validation Ensemble Results }\n")
        f.write("="*60 + "\n")
        f.write(f"OOF CV: R2 = {oof_r2:.4f}, RMSE = {oof_rmse:.4f}, MAE = {oof_mae:.4f}\n")
        f.write(f"5-Fold Ensemble Test: R2 = {ens_r2:.4f}, RMSE = {ens_rmse:.4f}, MAE = {ens_mae:.4f}\n")
        f.write("="*60 + "\n\n")
