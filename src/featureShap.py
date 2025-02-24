from captum.attr import ShapleyValueSampling
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
import random
from skorch import NeuralNetRegressor, NeuralNet
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse
from models.weaklearner import MLP_2HL
import pandas as pd

parser = argparse.ArgumentParser()

# Integer parameters with no default value and required flag
parser.add_argument('--feat_d', type=int, help='Feature dimension', default=19)  # 输入特征维度
parser.add_argument('--hidden_d', type=int, help='Hidden layer dimension', default=128)  # 弱学习器隐藏层维度

# Float parameters with no default value and required flag
parser.add_argument('--boost_rate', type=float, help='Boosting rate', default=1.0)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--L2', type=float, help='L2 regularization coefficient', default=1.0e-2)

# Integer parameters with default values
parser.add_argument('--num_nets', type=int, help='Number of networks', default=5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
parser.add_argument('--epochs_per_stage', type=int, help='Epochs per stage', default=100)
parser.add_argument('--correct_epoch', type=int, help='Epoch to correct model', default=100)

# String parameters with no default value and required flag
parser.add_argument('--data', type=str, help='Path to data')
parser.add_argument('--tr', type=str, help='Path to training data')
parser.add_argument('--te', type=str, help='Path to testing data')
parser.add_argument('--out_f', type=str, help='Output file path', default='../checkpoint/best_GrowNN.pth')

# Float parameter with default value


# Boolean flags
parser.add_argument('--sparse', action='store_true', help='Use sparse representation')
parser.add_argument('--normalization', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Enable normalization (true/false)')
parser.add_argument('--cv', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Enable cross-validation (true/false)')
parser.add_argument('--cuda', action='store_true', help='Use CUDA for GPU acceleration')

args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积结果一致
    torch.backends.cudnn.benchmark = False  # 禁用自动优化


set_seed(41)  # 设置全局种子
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"训练设备：{device}")
# mix_seed(9204)
# 数据路径
file_path = "../data/processed/MemTrOC-Dataset.csv"
data = pd.read_csv(file_path)

# 提取特征和标签
X = data.iloc[:, 4:23].values  # 特征（19维）
y = data.iloc[:, 23].values  # 标签

# 先划分数据集再进行归一化（关键修改！）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=41)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2 / 0.9, random_state=41)
N = len(X_train)
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

from models.ensemblemodel import DynamicNet

mymodel = DynamicNet.from_file(args.out_f, lambda best_stage: MLP_2HL.get_model(best_stage, args))

mymodel.to_eval()

plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Times new Roman'

plt.rcParams['font.size'] = 13

# plt.subplots_adjust(left=0.35, right=1.0, top=0.88, bottom=0.15)

# [Pure water flux (L·m-2·h-1),Pressure (bar),pH,Temperature (oC),Filtration duration (h),TrOC concentration (mg/L),
# MW (Da),MWCO (Da),Min projection (nm),Max projection (nm),Molecular radius (nm),Pore radius (nm),pKa1 ,
# Zeta potential (mV),log Kow,Contact angle (°),Molecular charge,Charge product,log D ]

features = ["Pure water flux (L·m-2·h-1)", "Pressure (bar)", "pH", "Temperature (oC)", "Filtration duration (h)",
            "TrOC concentration (mg/L)", "MW (Da)", "MWCO (Da)", "Min projection (nm)", "Max projection (nm)",
            "Molecular radius (nm)", "Pore radius (nm)", "pKa1", "Zeta potential (mV)", "log Kow",
            "Contact angle (°)", "Molecular charge", "Charge product", "log D "]


def predict(x):
    # 将输入数据转换为 PyTorch Tensor
    x_torch = torch.tensor(x, dtype=torch.float32).to(device)
    # 确保模型在评估模式下运行
    mymodel.to_eval
    # 获取模型输出
    with torch.no_grad():
        _, pred = mymodel.forward(x_torch)
    # 返回 NumPy 数组
    return pred.cpu().numpy().reshape(-1, 1)


# 创建 SHAP 解释器
explainer = shap.KernelExplainer(predict, X_train[:100])  # 使用部分训练数据作为背景数据
shap_values = explainer.shap_values(X_train[:100])  # 计算 SHAP 值，这里仅计算前 100 个测试样本

# 绘制 SHAP 总结图
shap.summary_plot(shap_values, X_train[:100], feature_names=features, show=False)

# 保存或显示图像
plt.tight_layout()
# plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# captumshap = ShapleyValueSampling(mymodel)
# print(captumshap)
# attributions = captumshap.attribute(X_train_t, n_samples=300)  # 200
# print(X_train)
# print(attributions)
# attributions_numpy = attributions.squeeze().cpu().detach().numpy()
# # print(attributions_numpy)
# # shap.summary_plot(attributions_numpy, x_train, feature_names=features, plot_size=(9, 6.5))
# #
# #
# # # plt.savefig('featureShap.tif', format='tif')
# #
# #
# x_train_np = X_train.numpy()
# x_test_np = X_test.numpy()
# # shap.dependence_plot('MWCO(Da)', attributions_numpy, x_train_np, feature_names=features,
# #                      interaction_index='PFOS con (ppb)', show=False)
# # plt.xlabel('MWCO')
# # plt.show()
#
# shap.dependence_plot('MWCO(Da)', attributions_numpy, x_train_np, feature_names=features, interaction_index='Pore size(nm)')
#
# shap.dependence_plot('Temperature (˚C)', attributions_numpy, x_train_np, feature_names=features,
#                      interaction_index=None)
# shap.dependence_plot('Pressure (MPa)', attributions_numpy, x_train_np, feature_names=features,
#                      interaction_index=None)
# shap.dependence_plot('Pressure (MPa)', attributions_numpy, x_train_np, feature_names=features,
#                      interaction_index='water flux(LMH)')
#
# # 对于阳离子，可以根据实际情况选择最相关的进行比较
# shap.dependence_plot('Divalent cations (mmol/L)', attributions_numpy, x_train_np, feature_names=features,
#                      interaction_index=None)
#
# # 对于阳离子，可以根据实际情况选择最相关的进行比较
# shap.dependence_plot('Divalent cations (mmol/L)', attributions_numpy, x_train_np, feature_names=features,
#                      interaction_index='Monovalent cations (mmol/L)')


import shap

# # 确保输入数据是 PyTorch 张量并且形状正确
# # x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
# # x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)   # 添加通道维度
# # 构建 shap解释器
# x_train_np = x_train.detach().cpu().numpy()
# print(x_train.shape)
# explainer = shap.GradientExplainer(mymodel, x_train)
#
# # 计算测试集的shap值
#
# # shap_values = explainer.shap_values(x_train)
#
# # shap.summary_plot(shap_values, x_train, feature_names=features, plot_type="dot")
#
# shap_interaction_values = explainer.shap_interaction_values(x_train.numpy())
#
# shap.summary_plot(shap_interaction_values, x_train.numpy())


# 创建 shap.Explanation 对象

# explainer = shap.DeepExplainer(mymodel, x_train)
# # shap_values = explainer.shap_values(x_train)
#
# # shap.plots.heatmap(shap_values)
#
# shap_explanation = shap.Explanation(values=attributions_numpy,
#
#                                     base_values=explainer.expected_value,
#
#                                     data=x_train, feature_names=features)
#
# # 绘制热图
# # 计算每个实例的总 SHAP 值并获取排序索引
# order = np.argsort(attributions_numpy.sum(1))
# plt.rcParams['figure.figsize'] = (12, 10)
# plt.tight_layout()
# shap.plots.heatmap(shap_explanation, instance_order=order)
