from dataset.dataset import TableGraphDataset
from src.utils.smiles2graph import create_graph_data_from_smiles
from torch_geometric.data import Batch
import argparse
import time
from models.weaklearner import MLP_GNN
from models.ensemblemodel import DynamicNetForMLPGNN
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
from shap import KernelExplainer
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import random
from captum.attr import ShapleyValueSampling
import shap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""
Training pipeline for Tabular Descriptors + Molecular Graph Data

"""
parser = argparse.ArgumentParser()

# Integer parameters with no default value and required flag
parser.add_argument('--feat_d', type=int, help='Feature dimension', default=19)  # Input feature dimension
parser.add_argument('--hidden_d', type=int, help='Hidden layer dimension', default=128)  # Hidden layer dimension of weak learner

parser.add_argument('--table_dim_in', type=int, default=19)  # Tabular feature input dimension
parser.add_argument('--table_dim_hidden', type=int, default=128)  # Note: processed parameter
parser.add_argument('--gnn_input_dim', type=int, default=9)  # Node feature dimension
parser.add_argument('--out_dim', type=int, default=128)  # Output dimension for tabular and graph features
parser.add_argument('--gnn_hidden', type=int, default=128)  # Hidden dimension for GNN feature extractor
parser.add_argument('--combined_dim', type=int, default=128)  # Combined feature dimension after fusion
parser.add_argument('--dim_hidden1', type=int, default=128)  # Hidden layer dimension after feature fusion
parser.add_argument('--dim_hidden2', type=int, default=128)  # Hidden layer dimension after feature fusion

# Float parameters with no default value and required flag
parser.add_argument('--boost_rate', type=float, help='Boosting rate', default=1.0)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--L2', type=float, help='L2 regularization coefficient', default=1.0e-2)

# Integer parameters with default values
parser.add_argument('--num_nets', type=int, help='Number of networks', default=5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=256)
parser.add_argument('--epochs_per_stage', type=int, help='Epochs per stage', default=100)
parser.add_argument('--correct_epoch', type=int, help='Epoch to correct model', default=100)

# String parameters with no default value and required flag
parser.add_argument('--data', type=str, help='Path to data')
parser.add_argument('--tr', type=str, help='Path to training data')
parser.add_argument('--te', type=str, help='Path to testing data')
parser.add_argument('--out_f', type=str, help='Output file path',
                    default='../../checkpoint/best_GrowTableGraphNN_0606.pth')

# Float parameter with default value


# Boolean flags
parser.add_argument('--sparse', action='store_true', help='Use sparse representation')
parser.add_argument('--normalization', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Enable normalization (true/false)')
parser.add_argument('--cv', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Enable cross-validation (true/false)')
parser.add_argument('--cuda', action='store_true', help='Use CUDA for GPU acceleration', default=False)

args = parser.parse_args()

if not args.cuda:
    torch.set_num_threads(4)


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    # optimizer = SGD(params, lr, weight_decay=weight_decay)
    return optimizer


def root_mse(net_ensemble, loader):
    loss = 0
    total = 0

    for x, graph_data, y in loader:
        if args.cuda:
            x = x

        with torch.no_grad():
            _, out = net_ensemble.forward(x, graph_data)
        y = y.cpu().numpy().reshape(len(y), 1)
        out = out.cpu().numpy().reshape(len(y), 1)
        loss += mean_squared_error(y, out) * len(y)
        total += len(y)
    return np.sqrt(loss / total)


def init_gbnn(train):
    positive = negative = 0
    for i in range(len(train)):
        if train[i][1] > 0:
            positive += 1
        else:
            negative += 1
    blind_acc = max(positive, negative) / (positive + negative)
    print(f'Blind accuracy: {blind_acc}')
    # print(f'Blind Logloss: {blind_acc}')
    return float(np.log(positive / negative))


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


def my_collate(batch):
    """
      collate   Data   batch。
    """
    table_features, graph_data, labels = zip(*batch)
    # smiles_features   labels
    table_features = torch.stack(table_features, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    # torch_geometric.data.Batch   from_data_list   Data
    graph_data = Batch.from_data_list(graph_data)

    return table_features, graph_data, labels


# RESULTS
def get_predictions(net_ensemble, loader):
    net_ensemble.to_eval()  # Note: processed parameter
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, graph_data, y in loader:
            # Note: processed parameter
            if args.cuda:
                x = x.to(device)
                graph_data = graph_data.to(device)
                y = y.to(device)

            # Note: processed parameter
            _, preds = net_ensemble.forward(x, graph_data)

            # RESULTS
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    # RESULTS
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def worker_init_fn(worker_id):
    np.random.seed(41 + worker_id)
    random.seed(41 + worker_id)


def set_seed(seed):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)  # Note: processed parameter
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # CUDA RESULTS
    torch.backends.cudnn.benchmark = False  # Note: processed parameter
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# 1.
class EnsembleModelWrapper(nn.Module):
    def __init__(self, ensemble_model, fixed_graph_data):
        """
        ensemble_model:  
        fixed_graph_data:  （ ）
        """
        super(EnsembleModelWrapper, self).__init__()
        self.ensemble_model = ensemble_model
        self.fixed_graph_data = fixed_graph_data

    def forward(self, x):
        """ ， """
        batch_size = x.shape[0]
        # batch
        graph_batch = Batch.from_data_list([self.fixed_graph_data] * batch_size)

        if args.cuda:
            x = x.to(device)
            graph_batch = graph_batch.to(device)

        with torch.no_grad():
            _, outputs = self.ensemble_model.forward(x, graph_batch)
        # _, outputs = self.ensemble_model.forward(x, graph_batch)
        # (batch_size, 1)
        # print("outputs:", outputs)
        # Note: processed parameter
        if outputs.dim() == 0:  # Note: processed parameter
            outputs = outputs.unsqueeze(0)
        if outputs.dim() == 1:  # Note: processed parameter
            outputs = outputs.unsqueeze(1)
        # print("after outputs.unsqueeze(1) outputs:", outputs)
        return outputs


# 2.  SHAP
def prepare_background_data(loader, sample_size=100):
    """ /Validation Set """
    background_table = []
    for x, _, _ in loader:
        background_table.append(x.cpu().numpy())
        if sum(len(arr) for arr in background_table) >= sample_size:
            break
    return np.vstack(background_table)[:sample_size]


def model_predict(x_array):
    """  numpy   tensor  """
    x_tensor = torch.tensor(x_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = wrapped_model(x_tensor)
    return outputs.cpu().numpy()


if __name__ == "__main__":
    set_seed(41)  # Set global random seed
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f" ：{device}")
    # Note: processed parameter
    file_path = "../../data/processed/MemTrOC-Dataset.csv"
    data = pd.read_csv(file_path)
    # data = data.head(10)
    # Note: processed parameter
    X = data.iloc[:, 4:23].values  # 19
    y = data.iloc[:, 23].values  # Note: processed parameter

    smiles_list = data.iloc[:, 3].values  # 3 SMILES

    # Note: processed parameter
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles_list, test_size=0.1,
                                                                                   random_state=41)
    X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(X_train, y_train, smiles_train,
                                                                                test_size=0.2 / 0.8,
                                                                                random_state=41)  # 0.2 / 0.8
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

    # Create dataset
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

    loader = test_loader

    best_stage = 4
    net_ensemble = DynamicNetForMLPGNN.from_file(args.out_f, lambda best_stage: MLP_GNN.get_model(best_stage, args))

    net_ensemble.to_eval()

    # 3.  Test Set
    explain_sample_indices = np.random.choice(len(test_dataset), min(100, len(test_dataset)), replace=False)

    # Train Set
    background_data = prepare_background_data(train_loader, sample_size=50)
    background_tensor = torch.tensor(background_data, dtype=torch.float32)

    all_shap_values = []
    all_sample_features = []  # Value
    feature_names = data.columns[4:23].tolist()  # Note: processed parameter

    # Note: processed parameter
    pbar = tqdm(explain_sample_indices, desc="Calculating SHAP values")

    for idx in pbar:
        # Note: processed parameter
        table_feat, graph_data, _ = test_dataset[idx]
        # Value
        all_sample_features.append(table_feat.cpu().numpy())  # Note: processed parameter
        # Note: processed parameter
        wrapped_model = EnsembleModelWrapper(net_ensemble, graph_data).to(device)
        wrapped_model.eval()

        # KernelExplainer
        explainer = KernelExplainer(
            model=model_predict,
            data=background_data,  # numpy
            link="identity"  # Note: processed parameter
        )

        # SHAPValue
        sample_tensor = table_feat.unsqueeze(0).to(device)  # batch
        sample_array = sample_tensor.cpu().numpy()
        # print(f"Sample array shape: {sample_array.shape}")  #   (1, 19)

        shap_values = explainer.shap_values(sample_array)
        # print(f"SHAP values shape: {np.array(shap_values).shape}")  #   (1, 19)   (19,)
        # all_shap_values.append(shap_values[0])   #
        all_shap_values.append(shap_values[0].flatten())  # batch

    # numpy
    all_shap_values = np.array(all_shap_values)
    all_sample_features = np.array(all_sample_features)  # Note: processed parameter
    print(f"All SHAP values shape: {all_shap_values.shape}")  # (n_samples, 19)

    plt.rcParams['font.family'] = 'serif'

    plt.rcParams['font.serif'] = 'Times new Roman'

    plt.rcParams['font.size'] = 13
    # shap.summary_plot(all_shap_values, X, feature_names=feature_names, plot_type="dot")

    # 5.  Feature Importance
    # plt.figure(figsize=(12, 6))
    mean_abs_shap = np.abs(all_shap_values).mean(axis=0).flatten()  # Note: processed parameter

    # NaN
    if np.isnan(mean_abs_shap).any():
        mean_abs_shap = np.nan_to_num(mean_abs_shap)

    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    # #
    # plt.bar(
    #     range(len(feature_names)),
    # mean_abs_shap[sorted_idx],  #
    #     color='#1f77b4'
    # )
    # plt.xticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    # plt.title('Global Feature Importance (mean |SHAP value|)')
    # plt.xlabel('Features')
    # plt.ylabel('Average Impact on Model Output')
    # plt.tight_layout()
    # plt.savefig('global_feature_importance.png')
    # plt.show()

    # ====================  SHAP  ====================
    # print("all_shap_values:", all_shap_values)
    # print("all_sample_features:", all_sample_features)
    # Note: processed parameter
    if len(feature_names) != all_shap_values.shape[1]:
        print(f" :  Name ({len(feature_names)}) SHAPValue ({all_shap_values.shape[1]}) ")
        # Name
        feature_names = [f"Feature_{i}" for i in range(all_shap_values.shape[1])]
    plt.figure(figsize=(10, 8))
    plt.tight_layout()
    shap.summary_plot(
        all_shap_values, # (n_samples, n_features)
        all_sample_features, # (n_samples, n_features)
        feature_names=feature_names,
        plot_type = "dot",
        show=False,  # Note: processed parameter
        max_display=15,  # 15
        plot_size=(10, 8)  # Note: processed parameter
    )
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')  # DPI
    plt.close()  # Note: processed parameter


    # #
    # plt.figure(figsize=(12, 8))
    # shap.summary_plot(
    #     all_shap_values,
    #     all_sample_features,
    #     feature_names=feature_names,
    # plot_type="violin",  #
    #     show=False
    # )
    # plt.title("SHAP Value Distribution", fontsize=14)
    # plt.tight_layout()
    # plt.savefig('shap_summary_violin_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()


    # RESULTS
    train_pred, train_true = get_predictions(net_ensemble, train_loader)
    val_pred, val_true = get_predictions(net_ensemble, val_loader)
    test_pred, test_true = get_predictions(net_ensemble, test_loader)

    prediction_train = train_pred
    prediction_val = val_pred
    prediction_test = test_pred
    y_train = train_true
    y_val = val_true
    y_test = test_true
    # _, prediction_train = net_ensemble.forward(X_train_t)
    # _, prediction_val = net_ensemble.forward(X_val_t)
    # _, prediction_test = net_ensemble.forward(X_test_t)

    from sklearn.metrics import r2_score

    R2_train = r2_score(y_train, prediction_train)
    R2_val = r2_score(y_val, prediction_val)
    R2_test = r2_score(y_test, prediction_test)

    # R2_train = 1 - torch.mean((y_train - prediction_train) ** 2) / torch.mean(
    #     (y_train - torch.mean(y_train)) ** 2)
    # R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean(
    #     (y_val - torch.mean(y_val)) ** 2)
    # R2_test = 1 - torch.mean((y_test - prediction_test) ** 2) / torch.mean(
    #     (y_test - torch.mean(y_test)) ** 2)
    print("------------------------RESULTS------------------------")
    print(f'train: R2：{R2_train}\n')
    print(f'val: R2：{R2_val}\n')
    print(f'test: R2：{R2_test}\n')

    rmse_train = np.sqrt(mean_squared_error(y_train, prediction_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, prediction_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, prediction_test))
    print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train, prediction_train))}\n')
    print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val, prediction_val))}\n')
    print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test, prediction_test))}\n')

    # Save the trained model (optional)
    # torch.save(trained_model.state_dict(), 'checkpoint/1DGBCNN_model.pth')

    # Train Set、Validation Set Test Set MAE
    mae_train = mean_absolute_error(y_train, prediction_train)
    mae_val = mean_absolute_error(y_val, prediction_val)
    mae_test = mean_absolute_error(y_test, prediction_test)

    # #  Train Set、Validation Set Test Set MAPE
    # mape_train = mean_absolute_percentage_error(y_train.numpy(), prediction_train.detach().numpy())
    # mape_val = mean_absolute_percentage_error(y_val.numpy(), prediction_val.detach().numpy())
    # mape_test = mean_absolute_percentage_error(y_test.numpy(), prediction_test.detach().numpy())

    print(f'train: MAE：{mae_train}\n')
    print(f'val: MAE：{mae_val}\n')
    print(f'test: MAE：{mae_test}\n')
    # print(f'train: MAPE：{mape_train}\n')
    # print(f'val: MAPE：{mape_val}\n')
    # print(f'test: MAPE：{mape_test}\n')
