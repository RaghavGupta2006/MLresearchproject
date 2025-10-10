import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import argparse
import random
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ensemblemodel import DynamicNet, DynamicNetForMLPGNN, DynamicNetForMLPImage
from models.weaklearner import MLP_2HL, MLP_GNN, MLP_ResNet, MLP_Maccs
from dataset.dataset import TableGraphDataset, TabularImageDataset
from src.utils.smiles2graph import create_graph_data_from_smiles
from torch_geometric.data import Batch


# -------------------------- 1. 全局配置：固定种子 + 科研样式 --------------------------
def set_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_scientific_plt_style():
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'grid.linewidth': 0.8,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'scatter.marker': 'o'
    })


# 初始化配置
set_seed()
init_scientific_plt_style()


# -------------------------- 2. 通用工具函数 --------------------------
def sample_data(true_values, pred_values, sample_size=50, seed=41):
    true_values = true_values.flatten() if len(true_values.shape) > 1 else true_values
    pred_values = pred_values.flatten() if len(pred_values.shape) > 1 else pred_values

    if sample_size == 0 or len(true_values) <= sample_size:
        return true_values, pred_values

    np.random.seed(seed)
    indices = np.random.choice(len(true_values), sample_size, replace=False)
    return true_values[indices], pred_values[indices]


def get_unified_axis_limits(all_true_values):
    all_true_flat = []
    for true_val in all_true_values:
        all_true_flat.extend(true_val.flatten() if len(true_val.shape) > 1 else true_val)
    global_min = np.min(all_true_flat)
    global_max = np.max(all_true_flat)
    margin = (global_max - global_min) * 0.05
    return global_min - margin, global_max + margin


def plot_true_vs_pred(ax, true_train, pred_train, true_test, pred_test,
                      model_name, unified_limits, sample_size=50,
                      fixed_r2=None, fixed_rmse=None):  # 新增：接收固定R²和RMSE
    """修改：用固定指标替换计算值，文本框显示R²和RMSE；标题格式改为“(a) GrowNN(Table Only)”"""
    sampled_true_train, sampled_pred_train = sample_data(true_train, pred_train, sample_size)
    sampled_true_test, sampled_pred_test = sample_data(true_test, pred_test, sample_size)

    # 绘制散点图（保持不变）
    ax.scatter(sampled_true_train, sampled_pred_train,
               color='#2E86AB', alpha=0.7, label='Train', edgecolor='white', linewidth=0.3, s=30)
    ax.scatter(sampled_true_test, sampled_pred_test,
               color='#A23B72', alpha=0.7, label='Test', edgecolor='white', linewidth=0.3, s=30)

    # 绘制理想线（保持不变）
    x_min, x_max = unified_limits
    ax.plot([x_min, x_max], [x_min, x_max], 'k--', linewidth=1.2, alpha=0.8, label='Ideal (y=x)')

    # 关键修改：使用固定传入的R²和RMSE，不再计算
    if fixed_r2 is not None and fixed_rmse is not None:
        text_box = f'R² = {fixed_r2:.4f}\nRMSE = {fixed_rmse:.4f}'  # 按用户要求保留4位小数
    else:
        # 兼容逻辑：若未传固定值，仍计算（可选保留）
        r2 = r2_score(np.concatenate([sampled_true_train, sampled_true_test]),
                      np.concatenate([sampled_pred_train, sampled_pred_test]))
        rmse = np.sqrt(mean_squared_error(np.concatenate([sampled_true_train, sampled_true_test]),
                                          np.concatenate([sampled_pred_train, sampled_pred_test])))
        text_box = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'

    # 显示指标文本框（保持样式不变）
    ax.text(0.05, 0.95, text_box, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', alpha=0.8, edgecolor='gray'),
            fontsize=14, fontweight='bold')

    # 轴设置（保持不变）
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel('True Values(%)', labelpad=8)
    ax.set_ylabel('Predicted Values(%)', labelpad=8)

    # -------------------------- 核心修改：标题格式调整 --------------------------
    # 拆分“字母”和“模型名”（如“a GrowNN(Table Only)”拆为“a”和“GrowNN(Table Only)”）
    letter, model_title = model_name.split(' ', 1)  # split(' ', 1)：只按第一个空格拆分
    ax.set_title(f'({letter}) {model_title}', fontweight='bold', pad=10)  # 格式改为“(a) 模型名”

    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, axis='both')


# -------------------------- 3. 数据加载与模型工具函数 --------------------------
def my_collate(batch, device):
    """自定义collate函数：适配GPU/CPU，按模型设备移数据"""
    table_features, graph_data, labels = zip(*batch)
    table_features = torch.stack(table_features, dim=0).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)
    graph_data = Batch.from_data_list(graph_data).to(device)  # PyG数据移到设备
    return table_features, graph_data, labels


def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    fingerprints = MACCSkeys.GenMACCSKeys(mol)
    return np.array([int(bit) for bit in fingerprints.ToBitString()], dtype=np.float32)


def load_data():
    """仅使用原始训练集（不合并验证集）"""
    file_path = "../data/processed/MemTrOC-Dataset.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = pd.read_csv(file_path)
    X = data.iloc[:, 4:23].values  # 19维表格特征
    y = data.iloc[:, 23].values  # 标签
    smiles_list = data.iloc[:, 3].values  # SMILES

    # 第一步：分割「train+val 集合」与「test 集合」（test占比10%）
    X_train_val, X_test, y_train_val, y_test, smiles_train_val, smiles_test = train_test_split(
        X, y, smiles_list, test_size=0.1, random_state=41, shuffle=True)

    # 第二步：分割「原始train集合」与「val集合」（val占train+val的2/9）
    X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(
        X_train_val, y_train_val, smiles_train_val, test_size=0.2 / 0.9, random_state=41, shuffle=True)

    # 特征归一化：仅用原始训练集拟合
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    return {
        "table": {
            "train": torch.tensor(X_train_scaled, dtype=torch.float32),  # 原始训练集
            "test": torch.tensor(X_test_scaled, dtype=torch.float32)
        },
        "y": {
            "train": y_train.reshape(-1, 1),  # 原始训练集标签
            "test": y_test.reshape(-1, 1)
        },
        "smiles": {
            "train": smiles_train,  # 原始训练集SMILES
            "test": smiles_test
        },
        "scaler_X": scaler_X
    }


# -------------------------- 4. 模型专属参数生成 --------------------------
def get_model_specific_args(model_type):
    """根据模型类型返回专属超参数"""
    args = argparse.Namespace()
    # 公共基础参数
    args.boost_rate = 1.0
    args.lr = 0.001
    args.L2 = 0.01
    args.sparse = False
    args.normalization = False
    args.cv = True
    args.data = None
    args.tr = None
    args.te = None

    # 模型专属参数
    if model_type == "grow_nn":
        args.feat_d = 19
        args.hidden_d = 128
        args.num_nets = 5
        args.batch_size = 64
        args.epochs_per_stage = 100
        args.correct_epoch = 100
        args.out_f = "../checkpoint/best_GrowNN_0514.pth"
        args.cuda = False

    elif model_type == "table_graph":
        args.feat_d = 19
        args.hidden_d = 128
        args.num_nets = 3
        args.batch_size = 256
        args.epochs_per_stage = 100
        args.correct_epoch = 100
        args.out_f = "../checkpoint/best_GrowTableGraphNN_0225.pth"
        args.cuda = False
        # GNN专属参数
        args.table_dim_in = 19
        args.table_dim_hidden = 128
        args.gnn_input_dim = 9
        args.out_dim = 128
        args.gnn_hidden = 128
        args.combined_dim = 128
        args.dim_hidden1 = 128
        args.dim_hidden2 = 128

    elif model_type == "table_image":
        args.feat_d = 186  # 19表格+167图像
        args.hidden_d = 256
        args.num_nets = 5
        args.batch_size = 32
        args.epochs_per_stage = 100
        args.correct_epoch = 100
        args.out_f = "../checkpoint/best_GrowTableImage_0304.pth"
        args.cuda = True
        # 图像专属参数
        args.table_dim_in = 19
        args.table_dim_hidden = 128
        args.out_dim = 128
        args.combined_dim = 128
        args.dim_hidden1 = 128
        args.dim_hidden2 = 128

    elif model_type == "table_maccs":
        args.feat_d = 186  # 19表格+167MACCS
        args.hidden_d = 256
        args.num_nets = 5
        args.batch_size = 32
        args.epochs_per_stage = 100
        args.correct_epoch = 100
        args.out_f = "../checkpoint/best_GrowTableMACCS_0514.pth"
        args.cuda = False

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # GPU可用性验证
    if args.cuda and not torch.cuda.is_available():
        print(f"⚠️ GPU not available, forcing cuda=False for {model_type}")
        args.cuda = False

    return args


# -------------------------- 5. 模型加载与预测 --------------------------
def load_model(model_type):
    """按模型类型加载模型"""
    args = get_model_specific_args(model_type)
    if not os.path.exists(args.out_f):
        raise FileNotFoundError(f"Model checkpoint not found: {args.out_f}")

    # 加载对应模型
    if model_type == "grow_nn":
        return DynamicNet.from_file(args.out_f, lambda stage: MLP_2HL.get_model(stage, args)), args
    elif model_type == "table_graph":
        return DynamicNetForMLPGNN.from_file(args.out_f, lambda stage: MLP_GNN.get_model(stage, args)), args
    elif model_type == "table_image":
        return DynamicNetForMLPImage.from_file(args.out_f, lambda stage: MLP_ResNet.get_model(stage, args)), args
    elif model_type == "table_maccs":
        return DynamicNet.from_file(args.out_f, lambda stage: MLP_Maccs.get_model(stage, args)), args


def get_model_predictions(model_type, data_dict):
    """按模型专属参数进行预测"""
    model, args = load_model(model_type)
    model.to_eval()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # 模型移至对应设备
    if hasattr(model, 'to'):
        model = model.to(device)
    elif args.cuda:
        model.to_cuda()

    print(f"🔧 {model_type} using device: {device} (batch_size={args.batch_size})")

    # 按模型类型预测
    if model_type == "grow_nn":
        X_train_t = data_dict["table"]["train"].to(device)
        X_test_t = data_dict["table"]["test"].to(device)

        def batch_predict(x, batch_size):
            model.to_eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(x), batch_size):
                    x_batch = x[i:i + batch_size]
                    _, p = model.forward(x_batch)
                    preds.append(p.cpu())
            return torch.cat(preds).numpy()

        pred_train = batch_predict(X_train_t, args.batch_size)
        pred_test = batch_predict(X_test_t, args.batch_size)
        return pred_train, pred_test

    elif model_type == "table_graph":
        # 加载表格+图数据集
        train_dataset = TableGraphDataset(
            data_dict["table"]["train"], data_dict["smiles"]["train"],
            torch.tensor(data_dict["y"]["train"], dtype=torch.float32),
            create_graph_data_from_smiles
        )
        test_dataset = TableGraphDataset(
            data_dict["table"]["test"], data_dict["smiles"]["test"],
            torch.tensor(data_dict["y"]["test"], dtype=torch.float32),
            create_graph_data_from_smiles
        )

        # 数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda batch: my_collate(batch, device)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda batch: my_collate(batch, device)
        )

        # 预测
        pred_train, pred_test = [], []
        with torch.no_grad():
            for x, graph, y in train_loader:
                _, p = model.forward(x, graph)
                pred_train.append(p.cpu())
            for x, graph, y in test_loader:
                _, p = model.forward(x, graph)
                pred_test.append(p.cpu())
        return torch.cat(pred_train).numpy(), torch.cat(pred_test).numpy()

    elif model_type == "table_image":
        # 加载表格+图像数据集
        train_dataset = TabularImageDataset(
            data_dict["table"]["train"], data_dict["smiles"]["train"],
            torch.tensor(data_dict["y"]["train"], dtype=torch.float32)
        )
        test_dataset = TabularImageDataset(
            data_dict["table"]["test"], data_dict["smiles"]["test"],
            torch.tensor(data_dict["y"]["test"], dtype=torch.float32)
        )

        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # 预测
        pred_train, pred_test = [], []
        with torch.no_grad():
            for x, img, y in train_loader:
                x, img = x.to(device), img.to(device)
                _, p = model.forward(x, img)
                pred_train.append(p.cpu())
            for x, img, y in test_loader:
                x, img = x.to(device), img.to(device)
                _, p = model.forward(x, img)
                pred_test.append(p.cpu())
        return torch.cat(pred_train).numpy(), torch.cat(pred_test).numpy()

    elif model_type == "table_maccs":
        # 生成MACCS指纹
        maccs_train = np.array([smiles_to_maccs(s) for s in data_dict["smiles"]["train"]])
        maccs_test = np.array([smiles_to_maccs(s) for s in data_dict["smiles"]["test"]])

        # 拼接特征（19+167=186）
        X_train_maccs = np.hstack([data_dict["table"]["train"].numpy(), maccs_train])
        X_test_maccs = np.hstack([data_dict["table"]["test"].numpy(), maccs_test])

        # 归一化
        scaler_maccs = MinMaxScaler()
        X_train_maccs_scaled = scaler_maccs.fit_transform(X_train_maccs)
        X_test_maccs_scaled = scaler_maccs.transform(X_test_maccs)

        # 转换为Tensor
        X_train_t = torch.tensor(X_train_maccs_scaled, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test_maccs_scaled, dtype=torch.float32).to(device)

        # 预测
        def batch_predict(x, batch_size):
            model.to_eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(x), batch_size):
                    x_batch = x[i:i + batch_size]
                    _, p = model.forward(x_batch)
                    preds.append(p.cpu())
            return torch.cat(preds).numpy()

        pred_train = batch_predict(X_train_t, args.batch_size)
        pred_test = batch_predict(X_test_t, args.batch_size)
        return pred_train, pred_test


# -------------------------- 6. 主绘图函数（核心修改：匹配子图传入固定指标） --------------------------
def create_scatter_plots(sample_size=500):
    results_dir = '../results'
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {results_dir}")

    # 加载数据（原始训练集+测试集）
    print("📥 Loading and preprocessing data...")
    data_dict = load_data()
    y_train = data_dict["y"]["train"]
    y_test = data_dict["y"]["test"]

    # 关键：定义每个子图的固定R²和RMSE（按用户提供的值）
    fixed_metrics = {
        "grow_nn": {"r2": 0.8494, "rmse": 11.2594},  # a图：GrowNN
        "table_graph": {"r2": 0.9014, "rmse": 9.1118},  # b图：Table+Graph
        "table_image": {"r2": 0.8571, "rmse": 10.9668},  # c图：Table+Image
        "table_maccs": {"r2": 0.7918, "rmse": 13.2400}  # d图：Table+MACCS
    }

    # 模型列表（与固定指标的key对应）
    models = [
        ("a", "GrowNN (Table Only)", "grow_nn"),
        ("b", "Table+Graph", "table_graph"),
        ("c", "Table+Image", "table_image"),
        ("d", "Table+MACCS", "table_maccs")
    ]

    # 获取所有模型预测值（仅用于绘制散点，指标用固定值）
    print("\n🚀 Loading models and generating predictions...")
    all_preds = []
    for idx, model_name, model_type in models:
        try:
            pred_train, pred_test = get_model_predictions(model_type, data_dict)
            all_preds.append((pred_train, pred_test))
            print(f"✅ {model_name} prediction completed")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to process {model_name}: {str(e)}") from e

    # 统一轴范围
    unified_limits = get_unified_axis_limits([y_train, y_test])

    # 创建子图
    print("\n🎨 Generating scatter plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('True vs Predicted Values of Multimodal Models',
                 fontsize=16, fontweight='bold', y=0.98)

    # 绘制每个子图（传入对应模型的固定指标）
    for i, ((idx, model_name, model_type), (pred_train, pred_test)) in enumerate(zip(models, all_preds)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # 关键：从fixed_metrics中获取当前模型的固定R²和RMSE
        metrics = fixed_metrics[model_type]
        plot_true_vs_pred(
            ax=ax,
            true_train=y_train,
            pred_train=pred_train,
            true_test=y_test,
            pred_test=pred_test,
            model_name=f"{idx} {model_name}",  # 传入“a GrowNN(Table Only)”格式的名称
            unified_limits=unified_limits,
            sample_size=sample_size,
            fixed_r2=metrics["r2"],  # 传入固定R²
            fixed_rmse=metrics["rmse"]  # 传入固定RMSE
        )

    # 保存图像
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = os.path.join(results_dir, 'model_comparison_scatter.png')
    pdf_path = os.path.join(results_dir, 'model_comparison_scatter.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"\n🎉 Scatter plots saved to:\n  - {png_path}\n  - {pdf_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model comparison scatter plots with specific params")
    parser.add_argument('--sample_size', type=int, default=0, help='Samples per dataset (train/test, 0=all)')
    args = parser.parse_args()

    create_scatter_plots(sample_size=args.sample_size)