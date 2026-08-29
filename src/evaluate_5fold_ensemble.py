import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import TableGraphDataset
from src.utils.smiles2graph import create_graph_data_from_smiles
from src.utils.physics_features import extract_features_and_labels
from models.weaklearner import MLP_GNN
from models.ensemblemodel import DynamicNetForMLPGNN

class Args:
    table_dim_in = 24
    table_dim_hidden = 128
    gnn_input_dim = 9
    edge_dim = 3
    gnn_type = 'gine'
    out_dim = 128
    gnn_hidden = 128
    combined_dim = 128
    dim_hidden1 = 128
    dim_hidden2 = 128
    sparse = False
    cuda = False

args = Args()

def my_collate(batch):
    table_features, graph_data, labels = zip(*batch)
    table_features = torch.stack(table_features, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    graph_data = Batch.from_data_list(graph_data)
    return table_features, graph_data, labels

def evaluate_ensemble(k_folds=5, data_file=None, checkpoint_dir=None, use_physics=True):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_file is None:
        data_file = os.path.join(base_dir, "data", "processed", "MemTrOC-Dataset.csv")
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(base_dir, "checkpoint")

    df = pd.read_csv(data_file)
    X_all, y_all, smiles_all = extract_features_and_labels(df, use_physics=use_physics)
    args.table_dim_in = X_all.shape[1]

    _, X_test, _, y_test, _, smiles_test = train_test_split(
        X_all, y_all, smiles_all, test_size=0.1, random_state=41
    )

    # Normalize based on full dataset representation
    scaler = MinMaxScaler()
    scaler.fit(X_all)
    X_test_norm = scaler.transform(X_test)

    test_ds = TableGraphDataset(
        torch.tensor(X_test_norm, dtype=torch.float32),
        smiles_test,
        torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
        create_graph_data_from_smiles
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=my_collate)

    fold_predictions = []
    for k in range(k_folds):
        ckpt_path = os.path.join(checkpoint_dir, f"best_GINE_fold{k}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Fold checkpoint {ckpt_path} not found, skipping.")
            continue

        model = DynamicNetForMLPGNN.from_file(ckpt_path, lambda s: MLP_GNN.get_model(s, args))
        model.to_eval()

        preds = []
        with torch.no_grad():
            for x, graph_data, y in test_loader:
                _, pred = model.forward(x, graph_data)
                preds.append(pred.cpu())
        fold_pred = torch.cat(preds).numpy()
        fold_predictions.append(fold_pred)

        f_r2 = r2_score(y_test, fold_pred)
        f_rmse = np.sqrt(mean_squared_error(y_test, fold_pred))
        f_mae = mean_absolute_error(y_test, fold_pred)
        print(f"Fold {k+1} Checkpoint -> Test R2: {f_r2:.4f}, RMSE: {f_rmse:.4f}, MAE: {f_mae:.4f}")

    if fold_predictions:
        ensemble_pred = np.mean(fold_predictions, axis=0)
        ens_r2 = r2_score(y_test, ensemble_pred)
        ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ens_mae = mean_absolute_error(y_test, ensemble_pred)
        print("\n" + "="*50)
        print(f"[*] 5-Fold Ensemble Test R2:   {ens_r2:.4f}")
        print(f"[*] 5-Fold Ensemble Test RMSE: {ens_rmse:.4f}%")
        print(f"[*] 5-Fold Ensemble Test MAE:  {ens_mae:.4f}%")
        print("="*50)
        return ens_r2, ens_rmse, ens_mae
    return None

if __name__ == "__main__":
    evaluate_ensemble()
