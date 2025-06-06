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
用于 表格数据 + 分子图数据 训练

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


# 1. 创建模型包装器（解决双输入问题）
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
        # _, outputs = self.ensemble_model.forward(x, graph_batch)
        # 确保输出为二维张量 (batch_size, 1)
        # print("outputs:", outputs)
        # 确保输出为二维：添加维度处理
        if outputs.dim() == 0:  # 标量转一维
            outputs = outputs.unsqueeze(0)
        if outputs.dim() == 1:  # 一维转二维
            outputs = outputs.unsqueeze(1)
        # print("after outputs.unsqueeze(1) outputs:", outputs)
        return outputs


# 2. 准备背景数据集（用于SHAP计算）
def prepare_background_data(loader, sample_size=100):
    """从训练/验证集采样背景数据"""
    background_table = []
    for x, _, _ in loader:
        background_table.append(x.cpu().numpy())
        if sum(len(arr) for arr in background_table) >= sample_size:
            break
    return np.vstack(background_table)[:sample_size]


def model_predict(x_array):
    """将 numpy 输入转为 tensor 并预测"""
    x_tensor = torch.tensor(x_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = wrapped_model(x_tensor)
    return outputs.cpu().numpy()


if __name__ == "__main__":
    set_seed(41)  # 设置全局种子
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"训练设备：{device}")
    # 数据路径
    file_path = "../../data/processed/MemTrOC-Dataset.csv"
    data = pd.read_csv(file_path)
    # data = data.head(10)
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

    loader = test_loader

    best_stage = 4
    net_ensemble = DynamicNetForMLPGNN.from_file(args.out_f, lambda best_stage: MLP_GNN.get_model(best_stage, args))

    net_ensemble.to_eval()

    # 3. 选择解释样本（从测试集抽样）
    explain_sample_indices = np.random.choice(len(test_dataset), min(100, len(test_dataset)), replace=False)

    # 准备背景数据（训练集的一部分）
    background_data = prepare_background_data(train_loader, sample_size=50)
    background_tensor = torch.tensor(background_data, dtype=torch.float32)

    all_shap_values = []
    all_sample_features = []  # 新增：收集特征值
    feature_names = data.columns[4:23].tolist()  # 从原始数据获取特征名

    # 进度条（每个样本单独解释）
    pbar = tqdm(explain_sample_indices, desc="Calculating SHAP values")

    for idx in pbar:
        # 获取测试样本
        table_feat, graph_data, _ = test_dataset[idx]
        # 保存特征值
        all_sample_features.append(table_feat.cpu().numpy())  # 新增
        # 创建包装模型
        wrapped_model = EnsembleModelWrapper(net_ensemble, graph_data).to(device)
        wrapped_model.eval()

        # 初始化 KernelExplainer
        explainer = KernelExplainer(
            model=model_predict,
            data=background_data,  # 注意使用 numpy 格式的背景数据
            link="identity"  # 回归任务使用恒等链接函数
        )

        # 计算SHAP值
        sample_tensor = table_feat.unsqueeze(0).to(device)  # 添加batch维度
        sample_array = sample_tensor.cpu().numpy()
        # print(f"Sample array shape: {sample_array.shape}")  # 调试：样本形状应为 (1, 19)

        shap_values = explainer.shap_values(sample_array)
        # print(f"SHAP values shape: {np.array(shap_values).shape}")  # 调试：应为 (1, 19) 或 (19,)
        # all_shap_values.append(shap_values[0])   # 柱状图的
        all_shap_values.append(shap_values[0].flatten())  # 移除batch维度

    # 转换为numpy数组
    all_shap_values = np.array(all_shap_values)
    all_sample_features = np.array(all_sample_features)  # 新增
    print(f"All SHAP values shape: {all_shap_values.shape}")  # 调试：应为 (n_samples, 19)

    plt.rcParams['font.family'] = 'serif'

    plt.rcParams['font.serif'] = 'Times new Roman'

    plt.rcParams['font.size'] = 13
    # shap.summary_plot(all_shap_values, X, feature_names=feature_names, plot_type="dot")

    # 5. 绘制全局特征重要性
    # plt.figure(figsize=(12, 6))
    mean_abs_shap = np.abs(all_shap_values).mean(axis=0).flatten()  # 关键修复：扁平化

    # 检查NaN
    if np.isnan(mean_abs_shap).any():
        mean_abs_shap = np.nan_to_num(mean_abs_shap)

    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    # # 明确使用一维数组
    # plt.bar(
    #     range(len(feature_names)),
    #     mean_abs_shap[sorted_idx],  # 现在是一维数组
    #     color='#1f77b4'
    # )
    # plt.xticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    # plt.title('Global Feature Importance (mean |SHAP value|)')
    # plt.xlabel('Features')
    # plt.ylabel('Average Impact on Model Output')
    # plt.tight_layout()
    # plt.savefig('global_feature_importance.png')
    # plt.show()

    # ==================== 新增：绘制带正负影响的SHAP摘要图 ====================
    # print("all_shap_values:", all_shap_values)
    # print("all_sample_features:", all_sample_features)
    # 确保特征数量匹配
    if len(feature_names) != all_shap_values.shape[1]:
        print(f"警告: 特征名称数量({len(feature_names)})与SHAP值维度({all_shap_values.shape[1]})不匹配")
        # 如果特征数量不一致，使用通用名称
        feature_names = [f"Feature_{i}" for i in range(all_shap_values.shape[1])]
    plt.figure(figsize=(10, 8))
    plt.tight_layout()
    shap.summary_plot(
        all_shap_values, # 形状应为 (n_samples, n_features)
        all_sample_features, # 形状应为 (n_samples, n_features)
        feature_names=feature_names,
        plot_type = "dot",
        show=False,  # 不立即显示，以便保存
        max_display=15,  # 最多显示15个最重要的特征
        plot_size=(10, 8)  # 显式设置绘图尺寸
    )
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')  # 保存时控制DPI和边界
    plt.close()  # 关闭图形释放内存


    # # 可选：绘制小提琴图展示分布
    # plt.figure(figsize=(12, 8))
    # shap.summary_plot(
    #     all_shap_values,
    #     all_sample_features,
    #     feature_names=feature_names,
    #     plot_type="violin",  # 小提琴图展示分布
    #     show=False
    # )
    # plt.title("SHAP Value Distribution", fontsize=14)
    # plt.tight_layout()
    # plt.savefig('shap_summary_violin_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()


    # 获取所有预测结果
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
    print("------------------------结果------------------------")
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

    # 计算训练集、验证集和测试集上的MAE
    mae_train = mean_absolute_error(y_train, prediction_train)
    mae_val = mean_absolute_error(y_val, prediction_val)
    mae_test = mean_absolute_error(y_test, prediction_test)

    # # 计算训练集、验证集和测试集上的MAPE
    # mape_train = mean_absolute_percentage_error(y_train.numpy(), prediction_train.detach().numpy())
    # mape_val = mean_absolute_percentage_error(y_val.numpy(), prediction_val.detach().numpy())
    # mape_test = mean_absolute_percentage_error(y_test.numpy(), prediction_test.detach().numpy())

    print(f'train: MAE：{mae_train}\n')
    print(f'val: MAE：{mae_val}\n')
    print(f'test: MAE：{mae_test}\n')
    # print(f'train: MAPE：{mape_train}\n')
    # print(f'val: MAPE：{mape_val}\n')
    # print(f'test: MAPE：{mape_test}\n')
