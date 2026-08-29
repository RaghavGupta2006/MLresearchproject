"""
Utility script to compute RDKit molecular descriptors from monomer SMILES strings.
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

def getTPSA(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.TPSA(mol) if mol else None

def getMolLogP(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolLogP(mol) if mol else None

def getMolWt(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol else None

def getqed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.qed(mol) if mol else None

def getBertzCT(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.BertzCT(mol) if mol else None

def getNumHAcceptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.NumHAcceptors(mol) if mol else None

def getNumHDonors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.NumHDonors(mol) if mol else None
