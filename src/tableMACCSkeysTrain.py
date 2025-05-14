from dataset.dataset import TableGraphDataset
from src.utils.smiles2graph import create_graph_data_from_smiles
from torch_geometric.data import Batch
import argparse
import time
from models.weaklearner import MLP_2HL, MLP_Maccs
from models.ensemblemodel import DynamicNet
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
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import random

"""
用于 表格数据 + MACCS 分子指纹数据 训练

"""
parser = argparse.ArgumentParser()

# Integer parameters with no default value and required flag
parser.add_argument('--feat_d', type=int, help='Feature dimension', default=19 + 167)  # 输入特征维度
parser.add_argument('--hidden_d', type=int, help='Hidden layer dimension', default=256)  # 弱学习器隐藏层维度

# Float parameters with no default value and required flag
parser.add_argument('--boost_rate', type=float, help='Boosting rate', default=1.0)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--L2', type=float, help='L2 regularization coefficient', default=1.0e-2)  # 1.0e-2

# Integer parameters with default values
parser.add_argument('--num_nets', type=int, help='Number of networks', default=5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)   # 32
parser.add_argument('--epochs_per_stage', type=int, help='Epochs per stage', default=100)
parser.add_argument('--correct_epoch', type=int, help='Epoch to correct model', default=100)

# String parameters with no default value and required flag
parser.add_argument('--data', type=str, help='Path to data')
parser.add_argument('--tr', type=str, help='Path to training data')
parser.add_argument('--te', type=str, help='Path to testing data')
parser.add_argument('--out_f', type=str, help='Output file path',
                    default='../checkpoint/best_GrowTableMACCS_0514.pth')

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

    for x, y in loader:
        if args.cuda:
            x = x

        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        y = y.cpu().numpy().reshape(len(y), 1)
        out = out.cpu().numpy().reshape(len(y), 1)
        loss += mean_squared_error(y, out) * len(y)
        total += len(y)
    return np.sqrt(loss / total)


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


# 将SMILES转换为MACCS指纹（处理无效分子）
def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 如果SMILES无效，返回全0向量
        return np.zeros(167, dtype=np.float32)
    fingerprints = MACCSkeys.GenMACCSKeys(mol)
    # 将指纹转换为167维的0/1数组
    return np.array([int(bit) for bit in fingerprints.ToBitString()], dtype=np.float32)


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


if __name__ == "__main__":
    seed = 41
    set_seed(seed)  # 设置全局种子
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"训练设备：{device}")
    # 数据路径
    file_path = "../data/processed/MemTrOC-Dataset.csv"
    data = pd.read_csv(file_path)

    # data = data.head(10)
    # 提取特征和标签
    X = data.iloc[:, 4:23].values  # 特征（19维）
    y = data.iloc[:, 23].values  # 标签

    smiles_list = data.iloc[:, 3].values  # 第3列是SMILES

    # 为所有SMILES生成MACCS特征
    maccs_features = np.array([smiles_to_maccs(smiles) for smiles in smiles_list])

    # 检查是否有无效SMILES
    invalid_mask = (maccs_features.sum(axis=1) == 0)  # 全0表示无效
    if invalid_mask.any():
        print(f"警告：发现 {invalid_mask.sum()} 个无效SMILES，已自动填充全0指纹")
    # 合并数值特征和MACCS指纹
    X = np.hstack([X.astype(np.float32), maccs_features])  # 最终形状：(n_samples, 19+167)

    print("X type:", type(X))
    print("X shape:", X.shape)
    print("X:", X)

    # 先划分数据集再进行归一化
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2 / 0.9, random_state=seed)
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
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # 创建 DataLoader
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    N = len(X_train)
    print(type(args.lr))
    print(type(args.boost_rate))

    best_rmse = pow(10, 6)
    val_rmse = best_rmse
    best_stage = args.num_nets - 1
    c0 = y_train.mean()  # init_gbnn(train) # c0是训练集目标值的平均值
    net_ensemble = DynamicNet(c0, args.boost_rate)  # 初始化集成网络，由多个弱学习器组成
    loss_f1 = nn.MSELoss()
    loss_models = torch.zeros((args.num_nets, 3))
    for stage in range(args.num_nets):  # 几个弱学习器就几个循环
        t0 = time.time()
        # 如果是第一个弱学习器，隐藏层维度和后面的弱学习器不一样
        model = MLP_Maccs.get_model(stage, args)  # Initialize the model_k: f_k(x), multilayer perception v2
        if args.cuda:
            model

        optimizer = get_optim(model.parameters(), args.lr, args.L2)  # 获得优化器
        net_ensemble.to_train()  # Set the models in ensemble net to train mode
        stage_mdlloss = []  # 保存每一次循环的损失。每多一次循环就多一个弱学习器。
        for epoch in range(args.epochs_per_stage):  # 在每一次循环中训练当前的弱学习器，弱学习器去学习残差
            for i, (x, y) in enumerate(train_loader):

                if args.cuda:
                    x = x
                    y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
                middle_feat, out = net_ensemble.forward(x)  # 使用多个弱学习器组合的学习器去得到预测值以及倒数第二层输出
                out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
                grad_direction = -(out - y)  # 根据预测值得到残差

                _, out = model(x, middle_feat)  # 使用当前弱学习器去学习残差，x以及强学习器的倒数第二层作为输入
                out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
                loss = loss_f1(net_ensemble.boost_rate * out, grad_direction)  # T

                model.zero_grad()
                loss.backward()
                optimizer.step()  # 对当前弱学习器进行参数更新
                stage_mdlloss.append(loss.item() * len(y))

        net_ensemble.add(model)  # 当前弱学习器训练好后，加入集成模型中
        sml = np.sqrt(np.sum(stage_mdlloss) / N)  # 平均每个训练样本的损失

        # 上述训练完成后得到一个有多个弱学习器组成的强学习器，下面再使用训练数据对该强学习器进行微调，学习率降低
        lr_scaler = 3
        # fully-corrective step
        stage_loss = []
        if stage > 0:
            # Adjusting corrective step learning rate
            if stage % 15 == 0:
                # lr_scaler *= 2
                args.lr /= 2
                args.L2 /= 2
            optimizer = get_optim(net_ensemble.parameters(), args.lr / lr_scaler, args.L2)
            for _ in range(args.correct_epoch):
                stage_loss = []
                for i, (x, y) in enumerate(train_loader):
                    if args.cuda:
                        x, y = x, y.view(-1, 1)
                    _, out = net_ensemble.forward_grad(x)
                    out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)

                    loss = loss_f1(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    stage_loss.append(loss.item() * len(y))
        # print(net_ensemble.boost_rate)
        # store model
        elapsed_tr = time.time() - t0
        sl = 0
        if stage_loss != []:
            sl = np.sqrt(np.sum(stage_loss) / N)

        print(
            f'Stage - {stage}, training time: {elapsed_tr: .1f} sec, model RMSE loss: {sml: .5f}, Ensemble Net RMSE '
            f'Loss: {sl: .5f}')

        net_ensemble.to_file(args.out_f)
        net_ensemble = DynamicNet.from_file(args.out_f, lambda stage: MLP_Maccs.get_model(stage, args))

        if args.cuda:
            net_ensemble.to_cuda()
        net_ensemble.to_eval()  # Set the models in ensemble net to eval mode

        # Train
        tr_rmse = root_mse(net_ensemble, train_loader)
        if args.cv:
            val_rmse = root_mse(net_ensemble, val_loader)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_stage = stage

        te_rmse = root_mse(net_ensemble, test_loader)

        print(f'Stage: {stage}  RMSE@Train: {tr_rmse:.5f}, RMSE@Val: {val_rmse:.5f}, RMSE@Test: {te_rmse:.5f}')

        loss_models[stage, 0], loss_models[stage, 1] = tr_rmse, te_rmse

    tr_rmse, te_rmse = loss_models[best_stage, 0], loss_models[best_stage, 1]
    print(f'Best validation stage: {best_stage}  RMSE@Train: {tr_rmse:.5f}, final RMSE@Test: {te_rmse:.5f}')
    loss_models = loss_models.detach().cpu().numpy()
    fname = './results/' + 'rmse'
    np.savez(fname, rmse=loss_models, params=args)

    print("best_stage:", best_stage)
    net_ensemble = DynamicNet.from_file(args.out_f, lambda best_stage: MLP_Maccs.get_model(best_stage, args))

    _, prediction_train = net_ensemble.forward(X_train_t)
    _, prediction_val = net_ensemble.forward(X_val_t)
    _, prediction_test = net_ensemble.forward(X_test_t)

    from sklearn.metrics import r2_score

    R2_train = r2_score(y_train, prediction_train.detach().cpu().numpy())
    R2_val = r2_score(y_val, prediction_val.detach().cpu().numpy())
    R2_test = r2_score(y_test, prediction_test.detach().cpu().numpy())

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
    rmse_train = np.sqrt(mean_squared_error(y_train, prediction_train.detach().cpu().numpy()))
    rmse_val = np.sqrt(mean_squared_error(y_val, prediction_val.detach().cpu().numpy()))
    rmse_test = np.sqrt(mean_squared_error(y_test, prediction_test.detach().cpu().numpy()))
    print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train, prediction_train.detach().numpy()))}\n')
    print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val, prediction_val.detach().numpy()))}\n')
    print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test, prediction_test.detach().numpy()))}\n')

    # Save the trained model (optional)
    # torch.save(trained_model.state_dict(), 'checkpoint/1DGBCNN_model.pth')

    # 计算训练集、验证集和测试集上的MAE
    mae_train = mean_absolute_error(y_train, prediction_train.detach().numpy())
    mae_val = mean_absolute_error(y_val, prediction_val.detach().numpy())
    mae_test = mean_absolute_error(y_test, prediction_test.detach().numpy())

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

    # 保存到日志文件
    argsDict = args.__dict__
    # 将结果和超参数保存到日志文件
    log_path = '../checkpoint/tableMACCSkeysTrain_log.txt'

    with open(log_path, 'a', encoding='utf-8') as f:  # 使用追加模式，并指定编码为utf-8
        # 添加文件顶部的分割线
        f.write("\n" + "=" * 60 + "\n")
        f.write("{" + "训练结果记录".center(58) + "}\n")
        f.write("=" * 60 + "\n\n")

        # 保存训练、验证、测试集的评价指标
        f.write("## 模型评估指标\n")
        f.write("-" * 60 + "\n")
        f.write("|       指标       |  训练集  |  验证集  |  测试集  |\n")
        f.write("-" * 60 + "\n")
        f.write(f"|     R² 值     | {R2_train:.4f}    | {R2_val:.4f}    | {R2_test:.4f}    |\n")
        f.write(f"|     RMSE     | {rmse_train:.4f}    | {rmse_val:.4f}    | {rmse_test:.4f}    |\n")
        f.write(f"|     MAE      | {mae_train:.4f}    | {mae_val:.4f}    | {mae_test:.4f}    |\n")
        f.write("-" * 60 + "\n\n")

        # 保存超参数
        f.write("## 超参数设置\n")
        f.write("-" * 60 + "\n")
        f.write("名称 | 值\n")
        f.write("-" * 60 + "\n")
        for eachArg, value in argsDict.items():
            f.write(f"{eachArg.ljust(20)} | {str(value).ljust(40)}\n")
        f.write("-" * 60 + "\n\n")

        # 文件底部的分割线
        f.write("=" * 60 + "\n\n")
