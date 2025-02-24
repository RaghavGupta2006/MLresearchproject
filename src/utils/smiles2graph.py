"""
把SMILES转换为图数据
"""

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from ogb.utils import smiles2graph
import pandas as pd
from torch_geometric.utils import add_self_loops


def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    return [int(x == s) for s in permitted_list]


def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
                               'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge',
                               'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    if not hydrogens_implicit:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    atom_type_enc = one_hot_encoding(atom.GetSymbol(), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [atom.IsInRing()]
    is_aromatic_enc = [atom.GetIsAromatic()]
    atomic_mass_scaled = [(atom.GetMass() - 10.812) / 116.092]
    vdw_radius_scaled = [(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6]
    covalent_radius_scaled = [(Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76]
    atom_features = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                               "CHI_OTHER"])
        atom_features += chirality_type_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_features += n_hydrogens_enc

    return np.array(atom_features, dtype=np.float32)


def create_graph_data_from_smiles111(smiles_list, target_list):
    """
    将SMILES字符串列表和对应的目标标签列表转换为图数据对象列表。
    """
    data_list = []
    for smiles, target in zip(smiles_list, target_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue  # 如果SMILES无效，则跳过

        n_atoms = mol.GetNumAtoms()
        atom_features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()])
        edge_index = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index += [[start, end], [end, start]]
        edge_index = torch.tensor(edge_index, dtype=torch.float).t().contiguous()
        target = torch.tensor([target], dtype=torch.float)

        data = Data(x=torch.tensor(atom_features, dtype=torch.float), edge_index=edge_index, y=target)
        # print("data:",data)
        data_list.append(data)

    return data_list


def create_graph_data_from_smiles(smiles, target):
    """
    将SMILES字符串列表和对应的目标标签列表转换为图数据对象列表。
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("mol无效")
        return
    # # 使用自定义的图数据
    # n_atoms = mol.GetNumAtoms()
    # 79维
    # atom_features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()])
    # edge_index = []
    # for bond in mol.GetBonds():
    #     start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    #     edge_index += [[start, end], [end, start]]
    # edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # target = torch.tensor([target], dtype=torch.float)
    # # print("atom_features:", atom_features)
    # # print("edge_index:", edge_index)
    # data = Data(x=torch.tensor(atom_features, dtype=torch.float), edge_index=edge_index, y=target)
    # # print("data:", data)

    # 使用 smiles2graph 库 获取图数据
    """
    原子类型（原子序数）
    手性（立体化学）
    连接度
    形式电荷
    连接的氢原子数
    自由基电子数
    杂化状态
    是否属于芳香环
    是否在环状结构中
    """
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat']).to(torch.float32)
    edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.float32)
    y = torch.tensor([target], dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

# 读取 CSV 文件，skiprows 用于跳过第一行
# df = pd.read_csv('BradleyMeltingPointDatasetClean.csv', skiprows=0)

# df = pd.read_csv('./data/bbbp.csv', skiprows=0)

# 定义要转换的SMILES字符串及其对应的目标值
# smiles_list = ['C1CCC(=CC1)CCN', 'CC(C)(C)OC(=O)N1CCC(CC1)OCC(=O)NC', 'CCCO', 'CCCCCCCCCCCCCCCO', 'CCCCCCCCN']
# target_list = [-55,95,86,58,-127,46,-1]
# smiles_list = df.iloc[:, 0]
# target_list = df.iloc[:, 1]

# # 转换SMILES到图数据
# graph_data_list = create_graph_data_from_smiles(smiles_list, target_list)
# print(len(graph_data_list))
# 将图数据保存为文件
# torch.save(graph_data_list, 'molecular_graphs_with_labels.pt')
