#!/usr/bin/env python3
"""
分子图特征重要性可视化分析
该脚本用于分析分子图在预测过程中各结构对预测结果的重要性，并通过颜色深浅直观表示。
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from enum import Enum
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所需的模型和工具函数
from src.utils.smiles2graph import create_graph_data_from_smiles
from models.weaklearner import MLP_GNN
from models.ensemblemodel import DynamicNetForMLPGNN
from torch_geometric.data import Batch

# 设置科研风格
plt.rcParams['font.family'] = 'Times New Roman'  # 使用更科研友好的字体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.transparent'] = False

# 定义常见官能团的SMARTS模式
COMMON_FUNCTIONAL_GROUPS = {
    'Hydroxyl (OH)': '[OX2H]',  # 羟基
    'Carboxylic Acid (COOH)': '[CX3](=O)[OX2H]',  # 羧基
    'Amine (NH2)': '[NX3;H2,H1;!$(NC=O)]',  # 氨基
    'Amide (CONH2)': '[NX3][CX3](=[OX1])[#6]',  # 酰胺基
    'Ester (COOR)': '[#6][CX3](=O)[OX2H0][#6]',  # 酯基
    'Ether (R-O-R)': '[OD2]([#6])[#6]',  # 醚基
    'Aldehyde (CHO)': '[CX3H1](=O)[#6]',  # 醛基
    'Ketone (CO)': '[#6][CX3](=O)[#6]',  # 酮基
    'Aromatic Amine': '[NX3;H2,H1;!$(NC=O);$(Nc1ccccc1)]',  # 芳香胺
    'Phenol': '[OX2H][cX3]:[cX3]',  # 酚羟基
    'Nitrile': '[NX1]#[CX2]',  # 腈基
    'Sulfonic Acid': '[SX4](=O)(=O)[OX2H]',  # 磺酸基
    'Carbonyl (C=O)': '[CX3]=[OX1]',  # 羰基
    'Methyl (CH3)': '[CX4H3]',  # 甲基
    'Ethyl (C2H5)': '[CX4]([#1])([#1])[CX4H3]',  # 乙基
    'Benzene Ring': 'c1ccccc1',  # 苯环
    'Fluoride (F)': '[F]',  # 氟
    'Chloride (Cl)': '[Cl]',  # 氯
    'Bromide (Br)': '[Br]',  # 溴
    'Iodide (I)': '[I]',  # 碘
    'Thiol (SH)': '[SX2H]',  # 巯基
    'Disulfide (S-S)': '[SX2][SX2]',  # 二硫键
    'Sulfoxide (S=O)': '[SX3](=O)([#6])[#6]',  # 亚砜
    'Sulfone (O=S=O)': '[SX4](=O)(=O)([#6])[#6]',  # 砜基
}


class MoleculeFeatureImportanceAnalyzer:
    """分子图特征重要性分析器"""

    def __init__(self, model_path, device=None):
        """初始化分析器

        Args:
            model_path: 模型权重文件路径
            device: 运行设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.model = self._load_model(model_path)
        # 使用模型自己的方法设置为评估模式，而不是标准PyTorch方法
        if hasattr(self.model, 'to_eval'):
            self.model.to_eval()
        else:
            self.model.eval()  # 备用方案
        if self.device.type == 'cuda':
            if hasattr(self.model, 'to_cuda'):
                self.model.to_cuda()
            else:
                self.model = self.model.to(self.device)  # 备用方案

    def _load_model(self, model_path):
        """加载训练好的模型"""
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 获取模型参数
        args = self._get_model_args()

        # 加载模型
        try:
            # 尝试使用PyTorch标准方式加载模型
            checkpoint = torch.load(model_path, map_location=self.device)

            # 检查checkpoint类型
            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                # 尝试从字典中获取模型
                model = checkpoint['model']
            else:
                # 使用DynamicNetForMLPGNN.from_file方法
                model = DynamicNetForMLPGNN.from_file(
                    model_path,
                    lambda stage: MLP_GNN.get_model(stage, args)
                )

            print(f"成功加载模型: {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _get_model_args(self):
        """获取模型所需的参数"""

        # 根据tableGraphTrainGPU.py中的参数设置
        class Args:
            def __init__(self):
                self.feat_d = 19
                self.hidden_d = 128
                self.num_nets = 3
                self.batch_size = 256
                self.epochs_per_stage = 100
                self.correct_epoch = 100
                self.table_dim_in = 19
                self.table_dim_hidden = 128
                self.gnn_input_dim = 9
                self.out_dim = 128
                self.gnn_hidden = 128
                self.combined_dim = 128
                self.dim_hidden1 = 128
                self.dim_hidden2 = 128
                self.boost_rate = 1.0
                self.lr = 0.001
                self.L2 = 0.01
                self.sparse = False
                self.normalization = False
                self.cv = True
                self.cuda = torch.cuda.is_available()

        return Args()

    def identify_functional_groups(self, mol):
        """识别分子中的官能团

        Args:
            mol: RDKit分子对象

        Returns:
            dict: 官能团名称到原子索引列表的映射
        """
        functional_groups = {}

        for fg_name, smarts_pattern in COMMON_FUNCTIONAL_GROUPS.items():
            try:
                pattern = Chem.MolFromSmarts(smarts_pattern)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    if matches:
                        # 收集所有匹配的原子索引
                        atom_indices = set()
                        for match in matches:
                            atom_indices.update(match)
                        functional_groups[fg_name] = list(atom_indices)
            except Exception as e:
                print(f"识别官能团 {fg_name} 时出错: {str(e)}")
                continue

        return functional_groups

    def compute_functional_group_importance(self, atom_importance, functional_groups):
        """计算官能团整体重要性

        Args:
            atom_importance: 原子重要性数组
            functional_groups: 官能团到原子索引的映射

        Returns:
            dict: 官能团名称到重要性分数的映射
        """
        fg_importance = {}

        for fg_name, atom_indices in functional_groups.items():
            # 计算官能团的整体重要性（可以使用多种聚合方法）
            # 1. 求和 - 体现官能团整体贡献
            sum_importance = sum(atom_importance[i] for i in atom_indices if i < len(atom_importance))

            # 2. 平均值 - 体现官能团平均贡献
            avg_importance = sum_importance / len(atom_indices) if atom_indices else 0

            # 3. 最大值 - 体现官能团中最关键原子的贡献
            max_importance = max([atom_importance[i] for i in atom_indices if i < len(atom_importance)], default=0)

            fg_importance[fg_name] = {
                'sum': sum_importance,
                'avg': avg_importance,
                'max': max_importance,
                'size': len(atom_indices),
                'atoms': atom_indices
            }

        return fg_importance

    def compute_atom_importance(self, table_data, smiles, target=None, method='gradient'):
        """计算分子中每个原子的重要性

        Args:
            table_data: 表格特征数据
            smiles: SMILES字符串
            target: 目标值
            method: 重要性计算方法

        Returns:
            atom_importance: 每个原子的重要性分数
            mol: RDKit分子对象
        """
        # 准备数据
        graph_data = create_graph_data_from_smiles(smiles, target or 0.0)
        if graph_data is None:
            raise ValueError(f"无效的SMILES字符串: {smiles}")

        # 转换为批处理格式
        # 确保table_data是数值类型
        try:
            # 尝试转换为float32类型
            if isinstance(table_data, np.ndarray):
                # 检查是否包含object类型
                if table_data.dtype == np.dtype('O'):
                    # 尝试转换所有元素为float
                    table_data = table_data.astype(np.float32)
            elif isinstance(table_data, (list, tuple)):
                # 列表或元组直接转换
                table_data = np.array(table_data, dtype=np.float32)

            table_tensor = torch.tensor(table_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"表格数据转换失败: {str(e)}")

        # 确保graph_data是张量类型并转移到正确设备
        if hasattr(graph_data, 'x') and isinstance(graph_data.x, np.ndarray):
            graph_data.x = torch.tensor(graph_data.x, dtype=torch.float32)

        graph_data = Batch.from_data_list([graph_data]).to(self.device)

        # 确保图数据节点特征需要梯度
        if hasattr(graph_data, 'x'):
            graph_data.x.requires_grad = True
        else:
            raise ValueError("图数据没有节点特征x")

        # 前向传播获取预测值 - 使用forward_grad方法以启用梯度计算
        try:
            if hasattr(self.model, 'forward_grad'):
                _, prediction = self.model.forward_grad(table_tensor, graph_data)
            else:
                # 如果没有forward_grad方法，使用普通forward但启用梯度
                with torch.enable_grad():
                    _, prediction = self.model.forward(table_tensor, graph_data)
        except Exception as e:
            raise RuntimeError(f"模型前向传播失败: {str(e)}")

        # 计算梯度重要性
        try:
            # 确保预测值是标量，如果不是则取均值或求和
            if prediction.numel() > 1:
                print(f"预测值形状: {prediction.shape}，将取均值进行反向传播")
                # 对于多输出模型，我们可以取均值或者只使用第一个输出
                prediction = prediction.mean()

            prediction.backward()
        except Exception as e:
            raise RuntimeError(f"梯度计算失败: {str(e)}")

        # 获取原子级别的梯度重要性
        if hasattr(graph_data.x, 'grad') and graph_data.x.grad is not None:
            # 添加调试信息查看梯度是否有效
            grad_sum = torch.sum(torch.abs(graph_data.x.grad)).item()
            print(f"梯度总和: {grad_sum}")

            # 使用梯度的绝对值或平方作为重要性指标
            if method == 'gradient':
                atom_importance = torch.abs(graph_data.x.grad).sum(dim=1).detach().cpu().numpy()
            elif method == 'gradient_squared':
                atom_importance = (graph_data.x.grad ** 2).sum(dim=1).detach().cpu().numpy()
            else:
                raise ValueError(f"不支持的重要性计算方法: {method}")

            # 打印重要性统计信息
            print(f"原子重要性范围: {np.min(atom_importance):.6f} - {np.max(atom_importance):.6f}")
        else:
            # 如果没有梯度，使用均匀分布的重要性
            print("警告: 未计算到有效梯度，使用均匀重要性")
            atom_importance = np.ones(graph_data.x.shape[0])

        # 获取RDKit分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法从SMILES创建RDKit分子对象: {smiles}")

        return atom_importance, mol

    def visualize_atom_importance(self, atom_importance, mol, smiles, figsize=(18, 6), dpi=300):
        """将原子重要性可视化到分子结构上，并添加关联图表

        Args:
            atom_importance: 每个原子的重要性分数
            mol: RDKit分子对象
            smiles: SMILES字符串
            figsize: 图像尺寸
            dpi: 图像分辨率

        Returns:
            fig: matplotlib图像对象
            ax: matplotlib坐标轴对象
        """
        # 打印重要性值的详细分布信息
        print(f"原子重要性详细分布 - 均值: {np.mean(atom_importance):.6f}, 标准差: {np.std(atom_importance):.6f}")
        print(f"重要性值排序: {sorted(atom_importance, reverse=True)[:5]}")  # 打印前5个最大的值

        # 归一化重要性分数以便于可视化
        # 使用更稳健的归一化方法，考虑分布情况
        if np.max(atom_importance) > 0:
            # 检查是否所有值都相同
            if np.all(atom_importance == atom_importance[0]):
                # 如果所有重要性值相同，强制使用均匀分布的颜色
                print("警告: 所有原子重要性值相同，将使用均匀颜色分布")
                atom_importance = np.linspace(0, 1, len(atom_importance))
                norm = Normalize(vmin=0, vmax=1)
            else:
                # 使用数据的实际范围进行归一化
                norm = Normalize(vmin=np.min(atom_importance), vmax=np.max(atom_importance))
        else:
            norm = Normalize(vmin=0, vmax=1)

        # 创建自定义颜色映射 - 从蓝色到深红色
        colors = ['#1f77b4', '#ffffff', '#8b0000']  # 蓝色 -> 白色 -> 深红色
        cmap = LinearSegmentedColormap.from_list('custom_red_blue', colors, N=256)

        # 创建科研风格的图形
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # 使用gridspec创建更灵活的布局 - 增加右侧子图宽度
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 3], wspace=0.4)  # 将右侧宽度比例从2增加到3
        ax1 = fig.add_subplot(gs[0, 0])  # 分子结构图
        ax2 = fig.add_subplot(gs[0, 1])  # 原子重要性条形图

        try:
            # 识别官能团
            functional_groups = self.identify_functional_groups(mol)
            # 计算官能团重要性
            fg_importance = self.compute_functional_group_importance(atom_importance, functional_groups)

            # 打印官能团重要性信息
            print(f"识别到 {len(functional_groups)} 个官能团:")
            for fg_name, imp in sorted(fg_importance.items(), key=lambda x: x[1]['sum'], reverse=True):
                print(f"  {fg_name}: 总重要性={imp['sum']:.4f}, 平均重要性={imp['avg']:.4f}, 原子数={imp['size']}")

            # 找出最重要的原子
            most_important_atom_idx = np.argmax(atom_importance) if len(atom_importance) > 0 else -1
            most_important_atom_value = atom_importance[most_important_atom_idx] if most_important_atom_idx >= 0 else 0
            print(f"最重要的原子索引: {most_important_atom_idx}, 重要性值: {most_important_atom_value:.4f}")

            # 左侧子图：分子结构可视化
            # 绘制分子结构，并根据原子重要性着色
            drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)

            # 设置原子颜色 - 使用自定义颜色映射
            atom_colors = {}
            for i, importance in enumerate(atom_importance):
                # 使用自定义颜色映射
                rgba = cmap(norm(importance))
                atom_colors[i] = (rgba[0], rgba[1], rgba[2])  # RGB，不含alpha

            # 准备绘制选项
            opts = Draw.MolDrawOptions()
            opts.padding = 0.2
            opts.useBWAtomPalette()

            # 收集所有需要高亮的原子和键
            highlight_atoms = list(range(len(atom_importance)))
            highlight_bonds = []

            # 找出包含最重要原子的所有官能团
            fg_with_most_important_atom = []
            if most_important_atom_idx >= 0:
                for fg_name, atoms in functional_groups.items():
                    if most_important_atom_idx in atoms:
                        fg_with_most_important_atom.append((fg_name, atoms))

                print(f"包含最重要原子的官能团: {[fg[0] for fg in fg_with_most_important_atom]}")

                # 如果找到了包含最重要原子的官能团，特殊高亮显示它们
                if fg_with_most_important_atom:
                    # 选择第一个包含最重要原子的官能团作为主要高亮目标
                    primary_fg_name, primary_fg_atoms = fg_with_most_important_atom[0]

                    # 找出官能团内的所有键
                    for bond in mol.GetBonds():
                        begin_atom_idx = bond.GetBeginAtomIdx()
                        end_atom_idx = bond.GetEndAtomIdx()
                        # 如果键连接的两个原子都在官能团内，高亮这个键
                        if begin_atom_idx in primary_fg_atoms and end_atom_idx in primary_fg_atoms:
                            highlight_bonds.append(bond.GetIdx())
                        # 扩大覆盖范围：如果键的一端在官能团内，另一端不在，也高亮这个键
                        elif begin_atom_idx in primary_fg_atoms or end_atom_idx in primary_fg_atoms:
                            highlight_bonds.append(bond.GetIdx())

            # 绘制分子，高亮所有原子和官能团内的键
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer, mol,
                highlightAtoms=highlight_atoms,
                highlightBonds=highlight_bonds,
                highlightAtomColors=atom_colors,
                highlightBondColors={b: (0.0, 1.0, 0.0, 0.5) for b in highlight_bonds}  # 半透明绿色高亮键
            )

            # 在分子图上添加官能团名称标注
            if fg_with_most_important_atom:
                try:
                    # 获取主要官能团的信息
                    primary_fg_name, primary_fg_atoms = fg_with_most_important_atom[0]

                    # 计算官能团的中心点位置用于放置标签
                    conf = mol.GetConformer()
                    if conf is not None and conf.IsValid() and len(primary_fg_atoms) > 0:
                        # 计算官能团中所有原子的平均坐标作为标签位置
                        x_coords = []
                        y_coords = []
                        for atom_idx in primary_fg_atoms:
                            if 0 <= atom_idx < mol.GetNumAtoms():
                                pos = conf.GetAtomPosition(atom_idx)
                                x_coords.append(pos.x)
                                y_coords.append(pos.y)

                        if x_coords and y_coords:
                            avg_x = sum(x_coords) / len(x_coords)
                            avg_y = sum(y_coords) / len(y_coords)

                            # 在分子图上添加官能团名称标签
                            text_x = int(avg_x * 40 + 400)
                            text_y = int(-avg_y * 40 + 300)

                            # 绘制半透明的蓝色背景框
                            text_width = len(primary_fg_name) * 12  # 估算每个字符宽度
                            drawer.SetFillColour((0.0, 0.8, 0.0, 0.6))  # RGBA
                            drawer.DrawRect(text_x - 5, text_y - 15, text_x + text_width + 5, text_y + 5)

                            # 绘制白色文本
                            drawer.DrawText(primary_fg_name, text_x, text_y, (1, 1, 1), fontScale=1.0)
                except Exception as e:
                    print(f"添加官能团标签时出错: {str(e)}")

            drawer.FinishDrawing()

            # 将Cairo图像转换为numpy数组
            import io
            img_data = drawer.GetDrawingText()
            img = plt.imread(io.BytesIO(img_data), format='png')

            # 显示分子图像
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title(f'Molecular Structure: {smiles}', fontsize=14, pad=15, fontweight='bold')

            # 右侧子图：科研风格的原子重要性条形图
            # 获取原子符号和索引
            atom_symbols_with_index = [f"{mol.GetAtomWithIdx(i).GetSymbol()}({i})" for i in range(mol.GetNumAtoms())]

            # 创建水平条形图，使用与分子图相同的颜色映射
            colors = [cmap(norm(imp)) for imp in atom_importance]
            bars = ax2.barh(range(len(atom_importance)), atom_importance, color=colors,
                            edgecolor='black', linewidth=0.5, alpha=0.8, height=0.7)

            # 突出显示最重要的原子 - 使用深红色
            if most_important_atom_idx >= 0:
                # 将最重要的原子设置为深红色
                bars[most_important_atom_idx].set_color('#8b0000')  # 深红色
                bars[most_important_atom_idx].set_edgecolor('darkred')
                bars[most_important_atom_idx].set_linewidth(3)
                bars[most_important_atom_idx].set_alpha(1.0)

                # 添加标注 - 调整箭头位置避免遮挡数字
                # 计算标注的位置
                x_pos = atom_importance[most_important_atom_idx]
                y_pos = most_important_atom_idx

                # 计算文本位置 - 将文本放在更右侧的位置
                text_x_offset = 0.25 * max(atom_importance) if max(atom_importance) > 0 else 0.1
                text_x = x_pos + text_x_offset

                # 如果文本位置超出当前x轴范围，调整x轴范围
                current_xlim = ax2.get_xlim()
                if text_x > current_xlim[1]:
                    ax2.set_xlim(current_xlim[0], text_x + 0.1 * max(atom_importance))

                # 添加标注，箭头从文本指向条形，避免遮挡数字
                ax2.annotate('Most Important',
                             xy=(x_pos, y_pos),  # 箭头指向的位置（条形末端）
                             xytext=(text_x, y_pos),  # 文本位置（更右侧）
                             arrowprops=dict(
                                 facecolor='darkred',
                                 shrink=0.05,
                                 width=1.5,
                                 headwidth=8,
                                 alpha=0.8,
                                 edgecolor='none'
                             ),
                             fontsize=10,
                             fontweight='bold',
                             color='darkred',
                             ha='left',  # 文本左对齐
                             va='center')

            # 设置坐标轴和标签
            ax2.set_yticks(range(len(atom_importance)))
            ax2.set_yticklabels(atom_symbols_with_index, fontsize=10)
            ax2.set_xlabel('Atom Importance Score', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Atom (Element & Index)', fontsize=12, fontweight='bold')
            ax2.set_title('Atom Importance Distribution', fontsize=14, pad=15, fontweight='bold')

            # 设置x轴范围，为标注留出空间
            x_max = max(atom_importance) * 1.3 if max(atom_importance) > 0 else 1.0
            ax2.set_xlim(0, x_max)

            # 添加网格线，使图表更易读
            ax2.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
            ax2.set_axisbelow(True)  # 将网格线放在条形后面

            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, atom_importance)):
                # 对于最重要的原子，使用白色文字提高可读性
                if i == most_important_atom_idx:
                    text_color = 'white'
                    fontweight = 'bold'
                else:
                    text_color = 'black'
                    fontweight = 'normal'

                # 调整文本位置，确保不会与箭头重叠
                text_x_pos = bar.get_width() + 0.01 * max(atom_importance)

                # 如果是重要原子且文本位置与箭头位置接近，稍微调整
                if i == most_important_atom_idx and text_x_pos > x_pos * 0.8:
                    text_x_pos = x_pos * 0.7  # 将文本放在条形内部

                ax2.text(text_x_pos,
                         bar.get_y() + bar.get_height() / 2,
                         f'{value:.4f}',
                         va='center', fontsize=9, fontweight=fontweight, color=text_color)

            # 添加共享的颜色条
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical',
                                fraction=0.03, pad=0.1, aspect=20)
            cbar.set_label('Importance Intensity', fontsize=11, fontweight='bold')
            cbar.ax.tick_params(labelsize=9)

            # 设置整体标题
            fig.suptitle('Molecular Feature Importance Analysis', fontsize=16, fontweight='bold', y=0.98)

        except Exception as e:
            # 如果RDKit绘制失败，使用备选方案
            ax1.text(0.5, 0.5, f'Molecular Visualization Failed: {str(e)}',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.axis('off')
            ax2.text(0.5, 0.5, f'Importance Chart Generation Failed: {str(e)}',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.axis('off')

        plt.tight_layout()
        return fig, ax1

    def analyze_and_visualize(self, table_data, smiles, target=None, output_path=None, method='gradient'):
        """分析分子特征重要性并可视化

        Args:
            table_data: 表格特征数据
            smiles: SMILES字符串
            target: 目标值
            output_path: 输出图像保存路径
            method: 重要性计算方法

        Returns:
            fig: matplotlib图像对象
        """
        # 计算原子重要性
        atom_importance, mol = self.compute_atom_importance(table_data, smiles, target, method)

        # 可视化
        fig, ax = self.visualize_atom_importance(atom_importance, mol, smiles)

        # 保存图像
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                        edgecolor='none', transparent=False)
            print(f"分子特征重要性图像已保存至: {output_path}")

        return fig

    def batch_analyze(self, data_file, output_dir, sample_size=None):
        """批量分析数据文件中的分子

        Args:
            data_file: 包含SMILES和表格特征的数据文件
            output_dir: 输出图像目录
            sample_size: 抽样数量，None表示分析所有数据
        """
        # 加载数据
        data = pd.read_csv(data_file)

        # 抽样
        if sample_size is not None and sample_size < len(data):
            data = data.sample(sample_size, random_state=41)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 对每个分子进行分析
        success_count = 0
        for i, row in data.iterrows():
            try:
                smiles = row.iloc[3]  # SMILES在第4列

                # 获取表格特征并确保是数值类型
                table_data = row.iloc[4:23].values  # 表格特征在第5-23列

                # 处理可能存在的非数值类型
                if isinstance(table_data, np.ndarray) and table_data.dtype == np.dtype('O'):
                    # 尝试将每个元素转换为float
                    table_data = np.array([self._safe_convert_to_float(x) for x in table_data])

                target = row.iloc[23]  # 目标值在第24列

                # 生成唯一的输出文件名
                safe_smiles = ''.join(c if c.isalnum() else '_' for c in smiles[:10])
                output_path = os.path.join(output_dir, f'molecule_importance_{i}_{safe_smiles}.png')

                # 分析并可视化
                self.analyze_and_visualize(table_data, smiles, target, output_path)
                success_count += 1

                # 每处理10个分子显示进度
                if success_count % 10 == 0:
                    print(f"已成功处理 {success_count} 个分子")

            except Exception as e:
                print(f"处理分子 {i} 失败: {str(e)}")
                continue

        print(f"批量分析完成，共尝试处理 {len(data)} 个分子，成功处理 {success_count} 个分子")

    def _safe_convert_to_float(self, value):
        """安全地将值转换为浮点数

        Args:
            value: 要转换的值

        Returns:
            float: 转换后的浮点数
        """
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0


# 示例用法
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="分子图特征重要性分析")
    parser.add_argument('--model_path', type=str, default='../checkpoint/best_GrowTableGraphNN_0225.pth',
                        help='模型权重文件路径')
    parser.add_argument('--data_file', type=str, default='../data/processed/MemTrOC-Dataset.csv',
                        help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='../results/molecule_importance',
                        help='输出图像保存目录')
    parser.add_argument('--sample_size', type=int, default=10, help='抽样分析的分子数量')
    parser.add_argument('--smiles', type=str, help='单个分析的SMILES字符串')
    parser.add_argument('--table_data', type=str, help='与SMILES对应的表格特征（逗号分隔）')

    args = parser.parse_args()

    # 初始化分析器
    analyzer = MoleculeFeatureImportanceAnalyzer(args.model_path)

    # 单分子分析或批量分析
    if args.smiles:
        # 单分子分析
        if args.table_data:
            table_data = np.array([float(x) for x in args.table_data.split(',')])
        else:
            # 如果没有提供表格特征，使用随机数据作为示例
            table_data = np.random.rand(19)
            print("警告：未提供表格特征，使用随机数据作为示例")

        fig = analyzer.analyze_and_visualize(
            table_data,
            args.smiles,
            output_path=os.path.join(args.output_dir, f'molecule_importance_{args.smiles[:10]}.png')
        )
        plt.show()
    else:
        # 批量分析
        analyzer.batch_analyze(args.data_file, args.output_dir, args.sample_size)