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
 SHAP ， Value SHAPValue 
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
parser.add_argument('--num_nets', type=int, help='Number of networks', default=3)
parser.add_argument('--batch_size', type=int, help='Batch size', default=256)
parser.add_argument('--epochs_per_stage', type=int, help='Epochs per stage', default=100)
parser.add_argument('--correct_epoch', type=int, help='Epoch to correct model', default=100)

# String parameters with no default value and required flag
parser.add_argument('--data', type=str, help='Path to data')
parser.add_argument('--tr', type=str, help='Path to training data')
parser.add_argument('--te', type=str, help='Path to testing data')
parser.add_argument('--out_f', type=str, help='Output file path',
                    default='../../checkpoint/best_GrowTableGraphNN_0606.pth')

# Note: processed parameter
parser.add_argument('--features', type=str, nargs='+', help='Features to plot dependence for',
                    default=['Molecular radius (nm)', 'MW (Da)', 'Pure water flux (L·m-2·h-1)', 'Pore radius (nm)',
                             'Filtration duration (h)', 'Molecular charge', 'log D '])

# Float parameter with default value
parser.add_argument('--sample_ratio', type=float, default=1.0, help='Ratio of test data to use (0.0-1.0)')

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


# Note: processed parameter
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
        # (batch_size, 1)
        if outputs.dim() == 0:  # Note: processed parameter
            outputs = outputs.unsqueeze(0)
        if outputs.dim() == 1:  # Note: processed parameter
            outputs = outputs.unsqueeze(1)
        return outputs


# SHAP
def prepare_background_data(loader, sample_size=100):
    """ /Validation Set """
    background_table = []
    for x, _, _ in loader:
        background_table.append(x.cpu().numpy())
        if sum(len(arr) for arr in background_table) >= sample_size:
            break
    return np.vstack(background_table)[:sample_size]


if __name__ == "__main__":
    set_seed(41)  # Set global random seed
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f" ：{device}")

    # Note: processed parameter
    file_path = "../../data/processed/MemTrOC-Dataset.csv"
    data = pd.read_csv(file_path)

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

    # Test Set
    X_test_original = X_test.copy()

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

    # Note: processed parameter
    best_stage = 4
    net_ensemble = DynamicNetForMLPGNN.from_file(args.out_f, lambda best_stage: MLP_GNN.get_model(best_stage, args))
    net_ensemble.to_eval()

    # Train Set
    background_data = prepare_background_data(train_loader, sample_size=50)

    # Note: processed parameter
    feature_names = data.columns[4:23].tolist()

    # Value SHAPValue
    all_sample_features = []  # Value SHAP
    all_sample_features_original = []  # Value
    all_shap_values = []

    # SHAPValue
    print(" SHAPValue， ...")

    # sample_ratio
    total_samples = len(test_dataset)
    num_samples = int(total_samples * args.sample_ratio)
    if num_samples < 1:  # 1
        num_samples = 1

    print(f"  {num_samples}/{total_samples}  SHAP ")

    # SHAPValue
    for idx in tqdm(range(num_samples), desc=" SHAPValue"):
        # Note: processed parameter
        table_feat, graph_data, _ = test_dataset[idx]

        # Value Value SHAP
        all_sample_features.append(table_feat.cpu().numpy())

        # Value
        original_feature_values = X_test_original[idx]  # Note: processed parameter
        all_sample_features_original.append(original_feature_values)

        # Note: processed parameter
        wrapped_model = EnsembleModelWrapper(net_ensemble, graph_data).to(device)
        wrapped_model.eval()


        # Note: processed parameter
        def model_predict(x_array):
            """  numpy   tensor  """
            x_tensor = torch.tensor(x_array, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = wrapped_model(x_tensor)
            return outputs.cpu().numpy()


        # KernelExplainer
        explainer = KernelExplainer(
            model=model_predict,
            data=background_data,  # numpy
            link="identity"  # Note: processed parameter
        )

        # SHAPValue
        sample_tensor = table_feat.unsqueeze(0).to(device)  # batch
        sample_array = sample_tensor.cpu().numpy()

        shap_values = explainer.shap_values(sample_array)
        all_shap_values.append(shap_values[0].flatten())  # batch

    # numpy
    all_shap_values = np.array(all_shap_values)
    all_sample_features = np.array(all_sample_features)
    all_sample_features_original = np.array(all_sample_features_original)

    print(f" : {all_sample_features_original.shape}")

    print(f" SHAPValue : {all_shap_values.shape}")
    print(f" : {all_sample_features.shape}")

    # Name
    if len(feature_names) != all_shap_values.shape[1]:
        print(f" :  Name ({len(feature_names)}) SHAPValue ({all_shap_values.shape[1]}) ")
        feature_names = [f"Feature_{i}" for i in range(all_shap_values.shape[1])]

    # ====================   ====================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'scatter.marker': 'o',
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3
    })

    # (2 3 )
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()  # 2x3 axes 1

    # -
    annotations = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # SHAP
    feature_contributions = {}

    # SHAP
    valid_features = []
    feature_data = []

    for feature_name in args.features:
        if feature_name in feature_names:
            # Note: processed parameter
            feature_idx = feature_names.index(feature_name)

            # SHAPValue
            feature_shap_values = all_shap_values[:, feature_idx]
            feature_values = all_sample_features_original[:, feature_idx]  # Value

            valid_features.append(feature_name)
            feature_data.append((feature_shap_values, feature_values, feature_idx))
        else:
            print(f" :   '{feature_name}'  ")

    # Note: processed parameter
    for i, (feature_name, (feature_shap_values, feature_values, feature_idx)) in enumerate(
            zip(valid_features, feature_data)):
        if i < len(axes):  # Note: processed parameter
            ax = axes[i]

            print(f"  '{feature_name}'  SHAP ...")

            # -
            scatter = ax.scatter(feature_values, feature_shap_values, alpha=0.9,
                                 c=feature_shap_values, cmap='coolwarm',
                                 vmin=-np.abs(feature_shap_values).max(),
                                 vmax=np.abs(feature_shap_values).max(),
                                 edgecolors='white', linewidth=0.5)

            # -
            z = np.polyfit(feature_values, feature_shap_values, 2)  # Note: processed parameter
            p = np.poly1d(z)
            x_trend = np.linspace(min(feature_values), max(feature_values), 100)
            ax.plot(x_trend, p(x_trend), '#1f77b4', linewidth=2.5, alpha=0.9, label='Trend line')

            # -
            ax.set_xlabel(feature_name, fontsize=14, fontweight='bold')
            ax.set_ylabel('SHAP Value', fontsize=14, fontweight='bold')

            # Note: processed parameter
            if i < len(annotations):
                ax.set_title(f'{annotations[i]} {feature_name}', fontsize=16, fontweight='bold', pad=15)
            else:
                ax.set_title(f'{feature_name}', fontsize=16, fontweight='bold', pad=15)

            # SHAPValue 0 -
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)

            # -
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            # Note: processed parameter
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('SHAP Value', fontsize=12)

            # Note: processed parameter
            ax.tick_params(axis='both', which='major', labelsize=11)

            # Note: processed parameter
            fig_single, ax_single = plt.subplots(figsize=(8, 6))

            # Note: processed parameter
            scatter_single = ax_single.scatter(feature_values, feature_shap_values, alpha=0.9,
                                               c=feature_shap_values, cmap='coolwarm',
                                               vmin=-np.abs(feature_shap_values).max(),
                                               vmax=np.abs(feature_shap_values).max(),
                                               edgecolors='white', linewidth=0.5)
            ax_single.plot(x_trend, p(x_trend), '#1f77b4', linewidth=2.5, alpha=0.9, label='Trend line')
            ax_single.set_xlabel(feature_name, fontsize=14, fontweight='bold')
            ax_single.set_ylabel('SHAP Value', fontsize=14, fontweight='bold')

            # Note: processed parameter
            if i < len(annotations):
                ax_single.set_title(f'{annotations[i]} {feature_name}', fontsize=16, fontweight='bold')
            else:
                ax_single.set_title(f'{feature_name}', fontsize=16, fontweight='bold')

            ax_single.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
            ax_single.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax_single.tick_params(axis='both', which='major', labelsize=11)

            # Note: processed parameter
            cbar_single = plt.colorbar(scatter_single, ax=ax_single)
            cbar_single.set_label('SHAP Value', fontsize=12)

            plt.tight_layout()
            plt.savefig(f'shap_dependence_{feature_name.replace(" ", "_").replace("/", "_")}.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig_single)

            print(f"SHAP : shap_dependence_{feature_name.replace(' ', '_').replace('/', '_')}.png")

            # Metric
            mean_abs_shap = np.abs(feature_shap_values).mean()
            std_shap = np.std(feature_shap_values)
            positive_ratio = np.mean(feature_shap_values > 0)
            negative_ratio = np.mean(feature_shap_values < 0)
            correlation = np.corrcoef(feature_values, feature_shap_values)[0, 1] if len(feature_values) > 1 else 0

            # RESULTS
            feature_contributions[feature_name] = {
                'mean_abs_shap': mean_abs_shap,
                'std_shap': std_shap,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'correlation': correlation
            }

            print(f"  '{feature_name}'  :")
            print(f"  -  SHAPValue: {mean_abs_shap:.4f}")
            print(f"  - SHAPValue : {std_shap:.4f}")
            print(f"  -  : {positive_ratio:.2%}")
            print(f"  -  : {negative_ratio:.2%}")
            print(f"  -  Value SHAPValue : {correlation:.4f}")
            print()

    # Note: processed parameter
    plt.figure(fig.number)
    plt.tight_layout(pad=3.0)  # Note: processed parameter
    plt.savefig('shap_dependence_plots_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig('shap_dependence_plots_combined.svg', bbox_inches='tight')  # SVG
    plt.close()
    print(" SHAP : shap_dependence_plots_combined.png")
    print(" SHAP : shap_dependence_plots_combined.svg")

    # CSV
    contributions_df = pd.DataFrame.from_dict(feature_contributions, orient='index')
    contributions_df.to_csv('feature_contributions_analysis.csv')
    print(" : feature_contributions_analysis.csv")

    # RESULTS
    print(" CSV SHAPValue 。")