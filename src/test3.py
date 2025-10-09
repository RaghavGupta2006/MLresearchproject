from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors
import pubchempy as pcp
import numpy as np


def generate_fingerprints(smiles):
    """根据SMILES字符串生成不同类型分子指纹，包括PubChem指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的SMILES字符串")

    # 1. MACCS 指纹 (167位)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_bits = ''.join(map(str, maccs_fp.ToList()))

    # 2. Morgan 指纹 (ECFP4变体，通常用半径=2)
    morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    morgan_bits = morgan_fp.ToBitString()

    # 3. RDKit拓扑指纹
    rdkit_fp = Chem.RDKFingerprint(mol)
    rdkit_bits = rdkit_fp.ToBitString()

    # 4. 原子对指纹
    atom_pair_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
    atom_pair_bits = atom_pair_fp.ToBitString()

    # 5. PubChem 指纹 (881位)
    pubchem_bits = None
    try:
        # 获取PubChem CID
        compounds = pcp.get_compounds(smiles, namespace='smiles')
        print("获取CID")
        print(compounds)
        if compounds:
            cid = compounds[0].cid

            # 获取指纹
            result = pcp.Compound.from_cid(cid).cactvs_fingerprint
            if result:
                # 将十六进制指纹转换为二进制数组
                hex_str = result.split()[0]
                n_bits = 881
                # 转换为二进制字符串并确保长度为881位
                binary_str = bin(int(hex_str, 16))[2:].zfill(n_bits)
                pubchem_bits = binary_str[-n_bits:]  # 取最后881位
    except Exception as e:
        print(f"生成PubChem指纹时出错: {str(e)}")
        pubchem_bits = '0' * 881  # 出错时返回全0

    # 如果无法获取PubChem指纹，则使用全0
    if pubchem_bits is None:
        pubchem_bits = '0' * 881

    return {
        "SMILES": smiles,
        "MACCS": maccs_bits,
        "Morgan": morgan_bits,
        "RDKit": rdkit_bits,
        "AtomPair": atom_pair_bits,
        "PubChem": pubchem_bits
    }


# 示例使用
if __name__ == "__main__":
    smiles_str = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # 咖啡因的SMILES

    try:
        fingerprints = generate_fingerprints(smiles_str)

        print(f"SMILES: {fingerprints['SMILES']}")
        print("\nMACCS 指纹 (167位):")
        print(fingerprints['MACCS'])

        print("\nMorgan 指纹 (ECFP4, 2048位):")
        print(fingerprints['Morgan'])

        print("\nRDKit 拓扑指纹 (2048位):")
        print(fingerprints['RDKit'])

        print("\n原子对指纹 (2048位):")
        print(fingerprints['AtomPair'])

        print("\nPubChem 指纹 (881位):")
        print(fingerprints['PubChem'])

        # 验证PubChem指纹长度
        print(f"\nPubChem指纹长度: {len(fingerprints['PubChem'])}")

    except ValueError as e:
        print(str(e))