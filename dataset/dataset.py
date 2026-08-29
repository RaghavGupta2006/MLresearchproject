import torch
from torch.utils.data import Dataset
from typing import Tuple, Any, Callable, Optional
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem import Draw


# Dataset for Tabular Data + Molecular Graph Data
class TableGraphDataset(Dataset):
    def __init__(self, tables, smiles, labels, create_graph_data_from_smiles, steric_ratios=None):
        """
        Initializes the TableGraphDataset class.

        Args:
            tables (torch.Tensor or np.ndarray): Tabular membrane descriptors.
            smiles (list of str): List containing SMILES strings.
            labels (list or torch.Tensor): Corresponding target rejection labels.
            create_graph_data_from_smiles (callable): Function to convert SMILES into PyG graph Data objects.
            steric_ratios (list or torch.Tensor, optional): Precomputed steric ratios for physics-constrained loss.
        """
        self.tables = tables
        self.smiles = smiles
        self.labels = labels
        self.create_graph_data_from_smiles = create_graph_data_from_smiles
        self.steric_ratios = steric_ratios

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx) -> Tuple[Any, ...]:
        table = self.tables[idx]
        smile = self.smiles[idx]
        label = self.labels[idx]

        # Generate molecular graph data from SMILES string
        graph_data = self.create_graph_data_from_smiles(str(smile), label)

        if self.steric_ratios is not None:
            steric = self.steric_ratios[idx]
            return table, graph_data, label, steric

        return table, graph_data, label


# Dataset for Tabular Data + 2D Molecular Image Data
class TabularImageDataset(Dataset):
    def __init__(self, X, smiles_list, y, image_transform=None):
        """
        Args:
            X (torch.Tensor or np.ndarray): Numerical tabular feature matrix.
            smiles_list (list): List of SMILES strings.
            y (torch.Tensor or np.ndarray): Target rejection labels.
            image_transform (torchvision.transforms, optional): Image preprocessing transformations.
        """
        self.X = X.clone().detach().to(torch.float32)
        self.smiles_list = smiles_list
        self.y = y.clone().detach().to(torch.float32)

        # Default image transformation: Resize, ToTensor, RGB Normalization
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        table_feature = self.X[idx]
        label = self.y[idx]

        # Convert SMILES to molecular image
        smiles = self.smiles_list[idx]
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {smiles}, index: {idx}")

        img = Draw.MolToImage(molecule, size=(224, 224))
        img = self.image_transform(img)

        return table_feature, img, label


# Dataset for Tabular Data + Molecular Graph + Molecular Image Data
class TableGraphImageDataset(Dataset):
    def __init__(
            self,
            tables,
            smiles_list,
            labels,
            create_graph_data_from_smiles,
            image_transform = None
    ):
        """
        Initializes dataset combining tabular descriptors, molecular graphs, and 2D images.

        Args:
            tables (torch.Tensor): Tabular feature tensor (num_samples x num_features).
            smiles_list (list): List of SMILES strings.
            labels (torch.Tensor): Target label tensor (num_samples x 1).
            create_graph_data_from_smiles (callable): Function to convert SMILES to PyG graph Data.
            image_transform (torchvision.transforms, optional): Preprocessing pipeline for images.
        """
        self.tables = tables.clone().detach().to(torch.float32)
        self.smiles_list = smiles_list
        self.labels = labels.clone().detach().to(torch.float32)
        self.create_graph_data = create_graph_data_from_smiles

        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx):
        table = self.tables[idx]
        smile = self.smiles_list[idx]
        label = self.labels[idx]

        # Generate molecular graph data
        graph_data = self.create_graph_data(str(smile), label)

        # Generate 2D molecular structure image
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smile}, index: {idx}")
        img = Draw.MolToImage(mol, size=(224, 224))
        img = self.image_transform(img)

        return table, graph_data, img, label
