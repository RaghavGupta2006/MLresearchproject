#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Feature Importance 
 analyze_single_molecule_importance Feature Importance
"""

import os
import sys
import numpy as np

# Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.molecule_feature_importance import (
    load_data, 
    load_model, 
    analyze_single_molecule_importance,
    set_seed
)


def analyze_custom_molecule(smiles):
    """ SMILES """
    print(f" : {smiles}")
    print("=" * 50)
    
    # Note: processed parameter
    set_seed(41)
    
    try:
        # 1.
        print(" 1:  ...")
        data_dict = load_data()
        print(f"✓   {len(data_dict['smiles']['train'])}  ")

        # 2.
        print("\n 2:  + ...")
        model, args, device = load_model()
        print(f"✓  ， : {device}")

        # 3.  RESULTS
        result_dir = '../results/molecule_feature_importance/single_molecule_custom'
        os.makedirs(result_dir, exist_ok=True)
        
        # 4.  Feature Importance
        print(f"\n 3:   {smiles}  Feature Importance...")
        atom_importance, feature_matrix = analyze_single_molecule_importance(
            smiles, model, data_dict, device, 
            save_path=f'{result_dir}/custom_molecule_analysis.png'
        )
        
        if atom_importance is not None:
            print(f"\n✓  !")
            print(f"  -  : {len(atom_importance)}")
            print(f"  -  : [{np.min(atom_importance):.6f}, {np.max(atom_importance):.6f}]")
            print(f"  -  : {np.mean(atom_importance):.6f}")
            if feature_matrix is not None:
                print(f"  -  : {feature_matrix.shape}")
            
            # ValueRESULTS
            np.save(f'{result_dir}/atom_importance.npy', atom_importance)
            if feature_matrix is not None:
                np.save(f'{result_dir}/feature_matrix.npy', feature_matrix)
            print(f"  - RESULTS : {result_dir}")
            
            return atom_importance, feature_matrix
        else:
            print(f"✗  !")
            return None, None
            
    except Exception as e:
        print(f"❌  : {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """ """
    print(" Feature Importance ")
    print("=" * 60)
    
    # SMILES
    example_smiles_list = [
        "CC(=O)Nc1ccc(O)cc1",  # Note: processed parameter
        "CC(C)C1CCC(C)CC1",    # Note: processed parameter
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Note: processed parameter
    ]
    
    # Note: processed parameter
    for i, smiles in enumerate(example_smiles_list, 1):
        print(f"\n{'='*20}   {i} {'='*20}")
        analyze_custom_molecule(smiles)
    
    print("\n" + "=" * 60)
    print("🎉  !")


if __name__ == "__main__":
    main()