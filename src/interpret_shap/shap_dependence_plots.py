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
用于绘制SHAP依赖图，分析特征值与SHAP值之间的关系
"""
parser = argparse.ArgumentParser()

# Integer parameters with no default value and required flag
parser.add_argument('--feat_d', type=int, help='Feature dimension', default=19)  # 输入特征维度
parser.add_argument('--hidden_d', type=int, help='Hidden layer dimension', default=128)  # 弱学习器隐藏层维度

parser.add_argument('--table_dim_in', type=int, default=19)  # 表格数据输入维度
parser.add_argument('--table_dim_hidden', type=int, default=128)  # 用于提取表格数据特征的网络隐藏层维度
parser.add_argument('--gnn_input_dim', type=int, default=9)  # 图数据 节点维度
parser.add_argument('--out_dim', type=int, default=128)  # 表格数据特征 与 图数据特征 的输出维度
parser.add_argument('--gnn_hidden', type=int, default=128)  # 用于提取图数据特征的网络隐藏层维度
parser.add_argument('--combined_dim', type=int, default=128)  # 表格数据特征 与 图数据特征 融合后的维度
parser.add_argument('--dim_hidden1', type=int, default=128)  # 特征融合后走的网络隐藏层维度
parser.add_argument('--dim_hidden2', type=int, default=128)  # 特征融合后走的网络隐藏层维度

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

# 指定要绘制依赖图的特征
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
    自定义 collate 函数来处理包含 Data 对象的 batch。
    """
    table_features, graph_data, labels = zip(*batch)
    # 将 smiles_features 和 labels 转换成张量
    table_features = torch.stack(table_features, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    # 使用 torch_geometric.data.Batch 的 from_data_list 方法来批量处理 Data 对象
    graph_data = Batch.from_data_list(graph_data)

    return table_features, graph_data, labels


# 定义收集预测结果的函数
def get_predictions(net_ensemble, loader):
    net_ensemble.to_eval()  # 切换到评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, graph_data, y in loader:
            # 移动数据到设备
            if args.cuda:
                x = x.to(device)
                graph_data = graph_data.to(device)
                y = y.to(device)

            # 执行预测
            _, preds = net_ensemble.forward(x, graph_data)

            # 收集结果
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    # 合并所有批次的预测结果
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def worker_init_fn(worker_id):
    np.random.seed(41 + worker_id)
    random.seed(41 + worker_id)


def set_seed(seed):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)  # 新增
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积结果一致
    torch.backends.cudnn.benchmark = False  # 禁用自动优化
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# 创建模型包装器（解决双输入问题）
class EnsembleModelWrapper(nn.Module):
    def __init__(self, ensemble_model, fixed_graph_data):
        """
        ensemble_model: 集成模型
        fixed_graph_data: 固定的分子图数据（单个样本）
        """
        super(EnsembleModelWrapper, self).__init__()
        self.ensemble_model = ensemble_model
        self.fixed_graph_data = fixed_graph_data

    def forward(self, x):
        """只接受表格输入，自动绑定固定的分子图"""
        batch_size = x.shape[0]
        # 复制固定图数据以匹配batch大小
        graph_batch = Batch.from_data_list([self.fixed_graph_data] * batch_size)

        if args.cuda:
            x = x.to(device)
            graph_batch = graph_batch.to(device)

        with torch.no_grad():
            _, outputs = self.ensemble_model.forward(x, graph_batch)
        # 确保输出为二维张量 (batch_size, 1)
        if outputs.dim() == 0:  # 标量转一维
            outputs = outputs.unsqueeze(0)
        if outputs.dim() == 1:  # 一维转二维
            outputs = outputs.unsqueeze(1)
        return outputs


# 准备背景数据集（用于SHAP计算）
def prepare_background_data(loader, sample_size=100):
    """从训练/验证集采样背景数据"""
    background_table = []
    for x, _, _ in loader:
        background_table.append(x.cpu().numpy())
        if sum(len(arr) for arr in background_table) >= sample_size:
            break
    return np.vstack(background_table)[:sample_size]


if __name__ == "__main__":
    set_seed(41)  # 设置全局种子
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    # 数据路径
    file_path = "../../data/processed/MemTrOC-Dataset.csv"
    data = pd.read_csv(file_path)

    # 提取特征和标签
    X = data.iloc[:, 4:23].values  # 特征（19维）
    y = data.iloc[:, 23].values  # 标签

    smiles_list = data.iloc[:, 3].values  # 第3列是SMILES

    # 先划分数据集再进行归一化
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles_list, test_size=0.1,
                                                                                   random_state=41)
    X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(X_train, y_train, smiles_train,
                                                                                test_size=0.2 / 0.8,
                                                                                random_state=41)  # 0.2 / 0.8

    # 保存原始的测试集特征数据用于绘图
    X_test_original = X_test.copy()

    # 只在训练集上拟合归一化器
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)  # 使用训练集的scaler
    X_test = scaler_X.transform(X_test)  # 使用训练集的scaler

    # 转换为 PyTorch Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 创建数据集
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

    # 加载模型
    best_stage = 4
    net_ensemble = DynamicNetForMLPGNN.from_file(args.out_f, lambda best_stage: MLP_GNN.get_model(best_stage, args))
    net_ensemble.to_eval()

    # 准备背景数据（训练集的一部分）
    background_data = prepare_background_data(train_loader, sample_size=50)

    # 获取特征名
    feature_names = data.columns[4:23].tolist()

    # 收集所有测试样本的特征值和SHAP值
    all_sample_features = []  # 归一化后的值，用于计算SHAP
    all_sample_features_original = []  # 原始值，用于绘图
    all_shap_values = []

    # 计算所有测试样本的SHAP值
    print("正在计算SHAP值，这可能需要一些时间...")

    # 根据sample_ratio决定要处理的样本数量
    total_samples = len(test_dataset)
    num_samples = int(total_samples * args.sample_ratio)
    if num_samples < 1:  # 确保至少处理1个样本
        num_samples = 1

    print(f"将使用 {num_samples}/{total_samples} 个测试样本进行SHAP分析")

    # 为部分或全部样本计算SHAP值
    for idx in tqdm(range(num_samples), desc="计算SHAP值"):
        # 获取测试样本
        table_feat, graph_data, _ = test_dataset[idx]

        # 保存特征值（归一化后的值，用于计算SHAP）
        all_sample_features.append(table_feat.cpu().numpy())

        # 保存原始特征值（用于绘图）
        original_feature_values = X_test_original[idx]  # 使用原始未归一化的测试数据
        all_sample_features_original.append(original_feature_values)

        # 创建包装模型
        wrapped_model = EnsembleModelWrapper(net_ensemble, graph_data).to(device)
        wrapped_model.eval()


        # 定义模型预测函数
        def model_predict(x_array):
            """将 numpy 输入转为 tensor 并预测"""
            x_tensor = torch.tensor(x_array, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = wrapped_model(x_tensor)
            return outputs.cpu().numpy()


        # 初始化 KernelExplainer
        explainer = KernelExplainer(
            model=model_predict,
            data=background_data,  # 注意使用 numpy 格式的背景数据
            link="identity"  # 回归任务使用恒等链接函数
        )

        # 计算SHAP值
        sample_tensor = table_feat.unsqueeze(0).to(device)  # 添加batch维度
        sample_array = sample_tensor.cpu().numpy()

        shap_values = explainer.shap_values(sample_array)
        all_shap_values.append(shap_values[0].flatten())  # 移除batch维度

    # 转换为numpy数组
    all_shap_values = np.array(all_shap_values)
    all_sample_features = np.array(all_sample_features)
    all_sample_features_original = np.array(all_sample_features_original)

    print(f"所有原始特征形状: {all_sample_features_original.shape}")

    print(f"所有SHAP值形状: {all_shap_values.shape}")
    print(f"所有样本特征形状: {all_sample_features.shape}")

    # 确保特征名称匹配
    if len(feature_names) != all_shap_values.shape[1]:
        print(f"警告: 特征名称数量({len(feature_names)})与SHAP值维度({all_shap_values.shape[1]})不匹配")
        feature_names = [f"Feature_{i}" for i in range(all_shap_values.shape[1])]

    # ==================== 设置科研风格的绘图参数 ====================
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

    # 创建包含六个子图的大图 (2行3列)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()  # 将2x3的axes数组展平为1维

    # 标注字符列表 - 现在将字母放在标题中
    annotations = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # 绘制指定特征的SHAP依赖图
    feature_contributions = {}

    # 收集所有特征的SHAP数据
    valid_features = []
    feature_data = []

    for feature_name in args.features:
        if feature_name in feature_names:
            # 获取特征索引
            feature_idx = feature_names.index(feature_name)

            # 计算此特征对所有样本的SHAP值
            feature_shap_values = all_shap_values[:, feature_idx]
            feature_values = all_sample_features_original[:, feature_idx]  # 使用原始值

            valid_features.append(feature_name)
            feature_data.append((feature_shap_values, feature_values, feature_idx))
        else:
            print(f"警告: 特征 '{feature_name}' 不在特征列表中")

    # 绘制子图
    for i, (feature_name, (feature_shap_values, feature_values, feature_idx)) in enumerate(
            zip(valid_features, feature_data)):
        if i < len(axes):  # 确保不超出子图数量
            ax = axes[i]

            print(f"绘制特征 '{feature_name}' 的SHAP依赖图...")

            # 绘制散点图 - 使用更专业的颜色和透明度
            scatter = ax.scatter(feature_values, feature_shap_values, alpha=0.9,
                                 c=feature_shap_values, cmap='coolwarm',
                                 vmin=-np.abs(feature_shap_values).max(),
                                 vmax=np.abs(feature_shap_values).max(),
                                 edgecolors='white', linewidth=0.5)

            # 添加趋势线 - 使用更明显的样式
            z = np.polyfit(feature_values, feature_shap_values, 2)  # 二次多项式拟合
            p = np.poly1d(z)
            x_trend = np.linspace(min(feature_values), max(feature_values), 100)
            ax.plot(x_trend, p(x_trend), '#1f77b4', linewidth=2.5, alpha=0.9, label='Trend line')

            # 添加标签和标题 - 将字母序号整合到标题中
            ax.set_xlabel(feature_name, fontsize=14, fontweight='bold')
            ax.set_ylabel('SHAP Value', fontsize=14, fontweight='bold')

            # 将字母序号放在标题中，更加显眼
            if i < len(annotations):
                ax.set_title(f'{annotations[i]} {feature_name}', fontsize=16, fontweight='bold', pad=15)
            else:
                ax.set_title(f'{feature_name}', fontsize=16, fontweight='bold', pad=15)

            # 添加水平线表示SHAP值为0 - 使用更明显的样式
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)

            # 添加网格 - 使用更细致的样式
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('SHAP Value', fontsize=12)

            # 设置坐标轴刻度更密集
            ax.tick_params(axis='both', which='major', labelsize=11)

            # 单独保存每个图表
            fig_single, ax_single = plt.subplots(figsize=(8, 6))

            # 应用相同的样式到单独图表
            scatter_single = ax_single.scatter(feature_values, feature_shap_values, alpha=0.9,
                                               c=feature_shap_values, cmap='coolwarm',
                                               vmin=-np.abs(feature_shap_values).max(),
                                               vmax=np.abs(feature_shap_values).max(),
                                               edgecolors='white', linewidth=0.5)
            ax_single.plot(x_trend, p(x_trend), '#1f77b4', linewidth=2.5, alpha=0.9, label='Trend line')
            ax_single.set_xlabel(feature_name, fontsize=14, fontweight='bold')
            ax_single.set_ylabel('SHAP Value', fontsize=14, fontweight='bold')

            # 单独图表的标题也包含字母序号
            if i < len(annotations):
                ax_single.set_title(f'{annotations[i]} {feature_name}', fontsize=16, fontweight='bold')
            else:
                ax_single.set_title(f'{feature_name}', fontsize=16, fontweight='bold')

            ax_single.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
            ax_single.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax_single.tick_params(axis='both', which='major', labelsize=11)

            # 为单独图表添加颜色条
            cbar_single = plt.colorbar(scatter_single, ax=ax_single)
            cbar_single.set_label('SHAP Value', fontsize=12)

            plt.tight_layout()
            plt.savefig(f'shap_dependence_{feature_name.replace(" ", "_").replace("/", "_")}.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig_single)

            print(f"SHAP依赖图已保存为: shap_dependence_{feature_name.replace(' ', '_').replace('/', '_')}.png")

            # 计算定量评估指标
            mean_abs_shap = np.abs(feature_shap_values).mean()
            std_shap = np.std(feature_shap_values)
            positive_ratio = np.mean(feature_shap_values > 0)
            negative_ratio = np.mean(feature_shap_values < 0)
            correlation = np.corrcoef(feature_values, feature_shap_values)[0, 1] if len(feature_values) > 1 else 0

            # 存储定量评估结果
            feature_contributions[feature_name] = {
                'mean_abs_shap': mean_abs_shap,
                'std_shap': std_shap,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'correlation': correlation
            }

            print(f"特征 '{feature_name}' 的定量评估数据:")
            print(f"  - 平均绝对SHAP值: {mean_abs_shap:.4f}")
            print(f"  - SHAP值标准差: {std_shap:.4f}")
            print(f"  - 正贡献比例: {positive_ratio:.2%}")
            print(f"  - 负贡献比例: {negative_ratio:.2%}")
            print(f"  - 特征值与SHAP值相关性: {correlation:.4f}")
            print()

    # 调整合并图的布局并保存
    plt.figure(fig.number)
    plt.tight_layout(pad=3.0)  # 增加子图间距
    plt.savefig('shap_dependence_plots_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig('shap_dependence_plots_combined.svg', bbox_inches='tight')  # 保存SVG格式
    plt.close()
    print("合并的SHAP依赖图已保存为: shap_dependence_plots_combined.png")
    print("合并的SHAP依赖图已保存为: shap_dependence_plots_combined.svg")

    # 保存所有特征的定量评估数据到CSV文件
    contributions_df = pd.DataFrame.from_dict(feature_contributions, orient='index')
    contributions_df.to_csv('feature_contributions_analysis.csv')
    print("特征贡献的定量评估数据已保存为: feature_contributions_analysis.csv")

    # 提示用户查看结果
    print("请查看保存的图片和CSV文件以分析特征与SHAP值之间的关系。")