#!/usr/bin/env python3
"""
 Feature Importance 
 RESULTS ， 。
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

# Note: processed parameter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Note: processed parameter
from src.utils.smiles2graph import create_graph_data_from_smiles
from models.weaklearner import MLP_GNN
from models.ensemblemodel import DynamicNetForMLPGNN
from torch_geometric.data import Batch

# Note: processed parameter
plt.rcParams['font.family'] = 'Times New Roman'  # Note: processed parameter
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.transparent'] = False

# SMARTS
COMMON_FUNCTIONAL_GROUPS = {
    'Hydroxyl (OH)': '[OX2H]',  # Note: processed parameter
    'Carboxylic Acid (COOH)': '[CX3](=O)[OX2H]',  # Note: processed parameter
    'Amine (NH2)': '[NX3;H2,H1;!$(NC=O)]',  # Note: processed parameter
    'Amide (CONH2)': '[NX3][CX3](=[OX1])[# 6]',  #
    'Ester (COOR)': '[# 6][CX3](=O)[OX2H0][#6]',  #
    'Ether (R-O-R)': '[OD2]([# 6])[#6]',  #
    'Aldehyde (CHO)': '[CX3H1](=O)[# 6]',  #
    'Ketone (CO)': '[# 6][CX3](=O)[#6]',  #
    'Aromatic Amine': '[NX3;H2,H1;!$(NC=O);$(Nc1ccccc1)]',  # Note: processed parameter
    'Phenol': '[OX2H][cX3]:[cX3]',  # Note: processed parameter
    'Nitrile': '[NX1]# [CX2]',  #
    'Sulfonic Acid': '[SX4](=O)(=O)[OX2H]',  # Note: processed parameter
    'Carbonyl (C=O)': '[CX3]=[OX1]',  # Note: processed parameter
    'Methyl (CH3)': '[CX4H3]',  # Note: processed parameter
    'Ethyl (C2H5)': '[CX4]([# 1])([#1])[CX4H3]',  #
    'Benzene Ring': 'c1ccccc1',  # Note: processed parameter
    'Fluoride (F)': '[F]',  # Note: processed parameter
    'Chloride (Cl)': '[Cl]',  # Note: processed parameter
    'Bromide (Br)': '[Br]',  # Note: processed parameter
    'Iodide (I)': '[I]',  # Note: processed parameter
    'Thiol (SH)': '[SX2H]',  # Note: processed parameter
    'Disulfide (S-S)': '[SX2][SX2]',  # Note: processed parameter
    'Sulfoxide (S=O)': '[SX3](=O)([# 6])[#6]',  #
    'Sulfone (O=S=O)': '[SX4](=O)(=O)([# 6])[#6]',  #
}


class MoleculeFeatureImportanceAnalyzer:
    """ Feature Importance """

    def __init__(self, model_path, device=None):
        """ 

        Args:
            model_path:  
            device:  
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" : {self.device}")
        self.model = self._load_model(model_path)
        # PyTorch
        if hasattr(self.model, 'to_eval'):
            self.model.to_eval()
        else:
            self.model.eval()  # Note: processed parameter
        if self.device.type == 'cuda':
            if hasattr(self.model, 'to_cuda'):
                self.model.to_cuda()
            else:
                self.model = self.model.to(self.device)  # Note: processed parameter

    def _load_model(self, model_path):
        """ """
        # Note: processed parameter
        if not os.path.exists(model_path):
            raise FileNotFoundError(f" : {model_path}")

        # Note: processed parameter
        args = self._get_model_args()

        # Note: processed parameter
        try:
            # PyTorch
            checkpoint = torch.load(model_path, map_location=self.device)

            # checkpoint
            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                # Note: processed parameter
                model = checkpoint['model']
            else:
                # DynamicNetForMLPGNN.from_file
                model = DynamicNetForMLPGNN.from_file(
                    model_path,
                    lambda stage: MLP_GNN.get_model(stage, args)
                )

            print(f" : {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f" : {str(e)}")

    def _get_model_args(self):
        """ """

        # tableGraphTrainGPU.py
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
        """ 

        Args:
            mol: RDKit 

        Returns:
            dict:  Name 
        """
        functional_groups = {}

        for fg_name, smarts_pattern in COMMON_FUNCTIONAL_GROUPS.items():
            try:
                pattern = Chem.MolFromSmarts(smarts_pattern)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    if matches:
                        # Note: processed parameter
                        atom_indices = set()
                        for match in matches:
                            atom_indices.update(match)
                        functional_groups[fg_name] = list(atom_indices)
            except Exception as e:
                print(f"  {fg_name}  : {str(e)}")
                continue

        return functional_groups

    def compute_functional_group_importance(self, atom_importance, functional_groups):
        """ 

        Args:
            atom_importance:  
            functional_groups:  

        Returns:
            dict:  Name 
        """
        fg_importance = {}

        for fg_name, atom_indices in functional_groups.items():
            # Note: processed parameter
            # 1.   -
            sum_importance = sum(atom_importance[i] for i in atom_indices if i < len(atom_importance))

            # 2.  Value -
            avg_importance = sum_importance / len(atom_indices) if atom_indices else 0

            # 3.  Value -
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
        """ 

        Args:
            table_data:  
            smiles: SMILES 
            target:  Value
            method:  

        Returns:
            atom_importance:  
            mol: RDKit 
        """
        # Note: processed parameter
        graph_data = create_graph_data_from_smiles(smiles, target or 0.0)
        if graph_data is None:
            raise ValueError(f"Invalid SMILES : {smiles}")

        # Note: processed parameter
        # table_data Value
        try:
            # float32
            if isinstance(table_data, np.ndarray):
                # object
                if table_data.dtype == np.dtype('O'):
                    # float
                    table_data = table_data.astype(np.float32)
            elif isinstance(table_data, (list, tuple)):
                # Note: processed parameter
                table_data = np.array(table_data, dtype=np.float32)

            table_tensor = torch.tensor(table_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f" : {str(e)}")

        # graph_data
        if hasattr(graph_data, 'x') and isinstance(graph_data.x, np.ndarray):
            graph_data.x = torch.tensor(graph_data.x, dtype=torch.float32)

        graph_data = Batch.from_data_list([graph_data]).to(self.device)

        # Note: processed parameter
        if hasattr(graph_data, 'x'):
            graph_data.x.requires_grad = True
        else:
            raise ValueError(" x")

        # Value -  forward_grad
        try:
            if hasattr(self.model, 'forward_grad'):
                _, prediction = self.model.forward_grad(table_tensor, graph_data)
            else:
                # forward_grad forward
                with torch.enable_grad():
                    _, prediction = self.model.forward(table_tensor, graph_data)
        except Exception as e:
            raise RuntimeError(f" : {str(e)}")

        # Note: processed parameter
        try:
            # Value Value
            if prediction.numel() > 1:
                print(f" Value : {prediction.shape}， Value ")
                # Value
                prediction = prediction.mean()

            prediction.backward()
        except Exception as e:
            raise RuntimeError(f" : {str(e)}")

        # Note: processed parameter
        if hasattr(graph_data.x, 'grad') and graph_data.x.grad is not None:
            # Note: processed parameter
            grad_sum = torch.sum(torch.abs(graph_data.x.grad)).item()
            print(f" : {grad_sum}")

            # Value Metric
            if method == 'gradient':
                atom_importance = torch.abs(graph_data.x.grad).sum(dim=1).detach().cpu().numpy()
            elif method == 'gradient_squared':
                atom_importance = (graph_data.x.grad ** 2).sum(dim=1).detach().cpu().numpy()
            else:
                raise ValueError(f" : {method}")

            # Note: processed parameter
            print(f" : {np.min(atom_importance):.6f} - {np.max(atom_importance):.6f}")
        else:
            # Note: processed parameter
            print(" :  ， ")
            atom_importance = np.ones(graph_data.x.shape[0])

        # RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f" SMILES RDKit : {smiles}")

        return atom_importance, mol

    def visualize_atom_importance(self, atom_importance, mol, smiles, figsize=(18, 6), dpi=300):
        """ ， 

        Args:
            atom_importance:  
            mol: RDKit 
            smiles: SMILES 
            figsize:  
            dpi:  

        Returns:
            fig: matplotlib 
            ax: matplotlib 
        """
        # Value
        print(f"  -  Value: {np.mean(atom_importance):.6f},  : {np.std(atom_importance):.6f}")
        print(f" Value : {sorted(atom_importance, reverse=True)[:5]}")  # 5 Value

        # Note: processed parameter
        # Note: processed parameter
        if np.max(atom_importance) > 0:
            # Value
            if np.all(atom_importance == atom_importance[0]):
                # Value
                print(" :  Value ， ")
                atom_importance = np.linspace(0, 1, len(atom_importance))
                norm = Normalize(vmin=0, vmax=1)
            else:
                # Note: processed parameter
                norm = Normalize(vmin=np.min(atom_importance), vmax=np.max(atom_importance))
        else:
            norm = Normalize(vmin=0, vmax=1)

        # -
        colors = ['# 1f77b4', '#ffffff', '#8b0000']  #   ->   ->
        cmap = LinearSegmentedColormap.from_list('custom_red_blue', colors, N=256)

        # Note: processed parameter
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # gridspec  -
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 3], wspace=0.4)  # 2 3
        ax1 = fig.add_subplot(gs[0, 0])  # Note: processed parameter
        ax2 = fig.add_subplot(gs[0, 1])  # Note: processed parameter

        try:
            # Note: processed parameter
            functional_groups = self.identify_functional_groups(mol)
            # Note: processed parameter
            fg_importance = self.compute_functional_group_importance(atom_importance, functional_groups)

            # Note: processed parameter
            print(f"  {len(functional_groups)}  :")
            for fg_name, imp in sorted(fg_importance.items(), key=lambda x: x[1]['sum'], reverse=True):
                print(f"  {fg_name}:  ={imp['sum']:.4f},  ={imp['avg']:.4f},  ={imp['size']}")

            # Note: processed parameter
            most_important_atom_idx = np.argmax(atom_importance) if len(atom_importance) > 0 else -1
            most_important_atom_value = atom_importance[most_important_atom_idx] if most_important_atom_idx >= 0 else 0
            print(f" : {most_important_atom_idx},  Value: {most_important_atom_value:.4f}")

            # Note: processed parameter
            # Note: processed parameter
            drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)

            # -
            atom_colors = {}
            for i, importance in enumerate(atom_importance):
                # Note: processed parameter
                rgba = cmap(norm(importance))
                atom_colors[i] = (rgba[0], rgba[1], rgba[2])  # RGB alpha

            # Note: processed parameter
            opts = Draw.MolDrawOptions()
            opts.padding = 0.2
            opts.useBWAtomPalette()

            # Note: processed parameter
            highlight_atoms = list(range(len(atom_importance)))
            highlight_bonds = []

            # Note: processed parameter
            fg_with_most_important_atom = []
            if most_important_atom_idx >= 0:
                for fg_name, atoms in functional_groups.items():
                    if most_important_atom_idx in atoms:
                        fg_with_most_important_atom.append((fg_name, atoms))

                print(f" : {[fg[0] for fg in fg_with_most_important_atom]}")

                # Note: processed parameter
                if fg_with_most_important_atom:
                    # Note: processed parameter
                    primary_fg_name, primary_fg_atoms = fg_with_most_important_atom[0]

                    # Note: processed parameter
                    for bond in mol.GetBonds():
                        begin_atom_idx = bond.GetBeginAtomIdx()
                        end_atom_idx = bond.GetEndAtomIdx()
                        # Note: processed parameter
                        if begin_atom_idx in primary_fg_atoms and end_atom_idx in primary_fg_atoms:
                            highlight_bonds.append(bond.GetIdx())
                        # Note: processed parameter
                        elif begin_atom_idx in primary_fg_atoms or end_atom_idx in primary_fg_atoms:
                            highlight_bonds.append(bond.GetIdx())

            # Note: processed parameter
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer, mol,
                highlightAtoms=highlight_atoms,
                highlightBonds=highlight_bonds,
                highlightAtomColors=atom_colors,
                highlightBondColors={b: (0.0, 1.0, 0.0, 0.5) for b in highlight_bonds}  # Note: processed parameter
            )

            # Name
            if fg_with_most_important_atom:
                try:
                    # Note: processed parameter
                    primary_fg_name, primary_fg_atoms = fg_with_most_important_atom[0]

                    # Note: processed parameter
                    conf = mol.GetConformer()
                    if conf is not None and conf.IsValid() and len(primary_fg_atoms) > 0:
                        # Note: processed parameter
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

                            # Name
                            text_x = int(avg_x * 40 + 400)
                            text_y = int(-avg_y * 40 + 300)

                            # Note: processed parameter
                            text_width = len(primary_fg_name) * 12  # Note: processed parameter
                            drawer.SetFillColour((0.0, 0.8, 0.0, 0.6))  # RGBA
                            drawer.DrawRect(text_x - 5, text_y - 15, text_x + text_width + 5, text_y + 5)

                            # Note: processed parameter
                            drawer.DrawText(primary_fg_name, text_x, text_y, (1, 1, 1), fontScale=1.0)
                except Exception as e:
                    print(f" : {str(e)}")

            drawer.FinishDrawing()

            # Cairo numpy
            import io
            img_data = drawer.GetDrawingText()
            img = plt.imread(io.BytesIO(img_data), format='png')

            # Note: processed parameter
            ax1.imshow(img)
            ax1.axis('off')
            # Note: processed parameter
            ax1.set_title(f'(a) Molecular Structure: {smiles}', fontsize=14, pad=15, fontweight='bold')

            # Note: processed parameter
            # Note: processed parameter
            atom_symbols_with_index = [f"{mol.GetAtomWithIdx(i).GetSymbol()}({i})" for i in range(mol.GetNumAtoms())]

            # Note: processed parameter
            colors = [cmap(norm(imp)) for imp in atom_importance]
            bars = ax2.barh(range(len(atom_importance)), atom_importance, color=colors,
                            edgecolor='black', linewidth=0.5, alpha=0.8, height=0.7)

            # -
            if most_important_atom_idx >= 0:
                # Note: processed parameter
                bars[most_important_atom_idx].set_color('# 8b0000')  #
                bars[most_important_atom_idx].set_edgecolor('darkred')
                bars[most_important_atom_idx].set_linewidth(3)
                bars[most_important_atom_idx].set_alpha(1.0)

                # -
                # Note: processed parameter
                x_pos = atom_importance[most_important_atom_idx]
                y_pos = most_important_atom_idx

                # -
                text_x_offset = 0.25 * max(atom_importance) if max(atom_importance) > 0 else 0.1
                text_x = x_pos + text_x_offset

                # x x
                current_xlim = ax2.get_xlim()
                if text_x > current_xlim[1]:
                    ax2.set_xlim(current_xlim[0], text_x + 0.1 * max(atom_importance))

                # Note: processed parameter
                ax2.annotate('Most Important',
                             xy=(x_pos, y_pos),  # Note: processed parameter
                             xytext=(text_x, y_pos),  # Note: processed parameter
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
                             ha='left',  # Note: processed parameter
                             va='center')

            # Note: processed parameter
            ax2.set_yticks(range(len(atom_importance)))
            ax2.set_yticklabels(atom_symbols_with_index, fontsize=10)
            ax2.set_xlabel('Atom Importance Score', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Atom (Element & Index)', fontsize=12, fontweight='bold')
            # Note: processed parameter
            ax2.set_title('(b) Atom Importance Distribution', fontsize=14, pad=15, fontweight='bold')

            # x
            x_max = max(atom_importance) * 1.3 if max(atom_importance) > 0 else 1.0
            ax2.set_xlim(0, x_max)

            # Note: processed parameter
            ax2.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
            ax2.set_axisbelow(True)  # Note: processed parameter

            # Value
            for i, (bar, value) in enumerate(zip(bars, atom_importance)):
                # Note: processed parameter
                if i == most_important_atom_idx:
                    text_color = 'white'
                    fontweight = 'bold'
                else:
                    text_color = 'black'
                    fontweight = 'normal'

                # Note: processed parameter
                text_x_pos = bar.get_width() + 0.01 * max(atom_importance)

                # Note: processed parameter
                if i == most_important_atom_idx and text_x_pos > x_pos * 0.8:
                    text_x_pos = x_pos * 0.7  # Note: processed parameter

                ax2.text(text_x_pos,
                         bar.get_y() + bar.get_height() / 2,
                         f'{value:.4f}',
                         va='center', fontsize=9, fontweight=fontweight, color=text_color)

            # Note: processed parameter
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical',
                                fraction=0.03, pad=0.1, aspect=20)
            cbar.set_label('Importance Intensity', fontsize=11, fontweight='bold')
            cbar.ax.tick_params(labelsize=9)

            # Note: processed parameter
            fig.suptitle('Molecular Feature Importance Analysis', fontsize=16, fontweight='bold', y=0.98)

        except Exception as e:
            # RDKit
            ax1.text(0.5, 0.5, f'Molecular Visualization Failed: {str(e)}',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.axis('off')
            ax2.text(0.5, 0.5, f'Importance Chart Generation Failed: {str(e)}',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.axis('off')

        plt.tight_layout()
        return fig, ax1

    def analyze_and_visualize(self, table_data, smiles, target=None, output_path=None, method='gradient'):
        """ Feature Importance 

        Args:
            table_data:  
            smiles: SMILES 
            target:  Value
            output_path:  
            method:  

        Returns:
            fig: matplotlib 
        """
        # Note: processed parameter
        atom_importance, mol = self.compute_atom_importance(table_data, smiles, target, method)

        # Note: processed parameter
        fig, ax = self.visualize_atom_importance(atom_importance, mol, smiles)

        # Note: processed parameter
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                        edgecolor='none', transparent=False)
            print(f" Feature Importance : {output_path}")
            
            # SVG
            svg_path = output_path.replace('.png', '.svg')
            fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white',
                        edgecolor='none', transparent=False)
            print(f" Feature ImportanceSVG : {svg_path}")

        return fig

    def batch_analyze(self, data_file, output_dir, sample_size=None):
        """ 

        Args:
            data_file:  SMILES 
            output_dir:  
            sample_size:  ，None 
        """
        # Note: processed parameter
        data = pd.read_csv(data_file)

        # Note: processed parameter
        if sample_size is not None and sample_size < len(data):
            data = data.sample(sample_size, random_state=41)

        # Note: processed parameter
        os.makedirs(output_dir, exist_ok=True)

        # Note: processed parameter
        success_count = 0
        for i, row in data.iterrows():
            try:
                smiles = row.iloc[3]  # SMILES 4

                # Value
                table_data = row.iloc[4:23].values  # 5-23

                # Value
                if isinstance(table_data, np.ndarray) and table_data.dtype == np.dtype('O'):
                    # float
                    table_data = np.array([self._safe_convert_to_float(x) for x in table_data])

                target = row.iloc[23]  # Value 24

                # Note: processed parameter
                safe_smiles = ''.join(c if c.isalnum() else '_' for c in smiles[:10])
                output_path = os.path.join(output_dir, f'molecule_importance_{i}_{safe_smiles}.png')

                # Note: processed parameter
                self.analyze_and_visualize(table_data, smiles, target, output_path)
                success_count += 1

                # 10
                if success_count % 10 == 0:
                    print(f"  {success_count}  ")

            except Exception as e:
                print(f"  {i}  : {str(e)}")
                continue

        print(f" ，  {len(data)}  ，  {success_count}  ")

    def _safe_convert_to_float(self, value):
        """ Value 

        Args:
            value:  Value

        Returns:
            float:  
        """
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0


# Note: processed parameter
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=" Feature Importance ")
    parser.add_argument('--model_path', type=str, default='../checkpoint/best_GrowTableGraphNN_0225.pth',
                        help=' ')
    parser.add_argument('--data_file', type=str, default='../data/processed/MemTrOC-Dataset.csv',
                        help=' ')
    parser.add_argument('--output_dir', type=str, default='../results/molecule_importance',
                        help=' ')
    parser.add_argument('--sample_size', type=int, default=10, help=' ')
    parser.add_argument('--smiles', type=str, help=' SMILES ')
    parser.add_argument('--table_data', type=str, help=' SMILES （ ）')

    args = parser.parse_args()

    # Note: processed parameter
    analyzer = MoleculeFeatureImportanceAnalyzer(args.model_path)

    # Note: processed parameter
    if args.smiles:
        # Note: processed parameter
        if args.table_data:
            table_data = np.array([float(x) for x in args.table_data.split(',')])
        else:
            # Note: processed parameter
            table_data = np.random.rand(19)
            print(" ： ， ")

        fig = analyzer.analyze_and_visualize(
            table_data,
            args.smiles,
            output_path=os.path.join(args.output_dir, f'molecule_importance_{args.smiles[:10]}.png')
        )
        plt.show()
    else:
        # Note: processed parameter
        analyzer.batch_analyze(args.data_file, args.output_dir, args.sample_size)