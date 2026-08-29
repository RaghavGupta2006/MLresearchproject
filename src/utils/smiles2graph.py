"""
Convert molecular SMILES strings into PyTorch Geometric graph Data objects.
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


def create_graph_data_from_smiles(smiles, target):
    """
    Converts a SMILES string and target label into a PyG Data object with 3D edge attributes.
    
    Node Features (9D from OGB):
    - Atomic number
    - Chirality
    - Degree
    - Formal charge
    - Number of Hydrogens
    - Number of radical electrons
    - Hybridization
    - Aromaticity
    - Ring membership
    
    Edge Features (3D from OGB):
    - Bond type (single, double, triple, aromatic)
    - Bond stereo
    - Is conjugated
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: Invalid SMILES string: {smiles}")
        return None

    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat']).to(torch.float32)
    edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.float32)
    y = torch.tensor([target], dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data
