#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单分子特征重要性分析使用示例
展示了如何使用analyze_single_molecule_importance函数分析特定分子的结构特征重要性
"""

import os
import sys
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.molecule_feature_importance import (
    load_data, 
    load_model, 
    analyze_single_molecule_importance,
    set_seed
)


def analyze_custom_molecule(smiles):
    """分析指定SMILES的分子"""
    print(f"开始分析分子: {smiles}")
    print("=" * 50)
    
    # 初始化设置
    set_seed(41)
    
    try:
        # 1. 加载数据
        print("步骤1: 加载数据...")
        data_dict = load_data()
        print(f"✓ 成功加载 {len(data_dict['smiles']['train'])} 个训练分子")

        # 2. 加载模型
        print("\n步骤2: 加载表格+分子图融合模型...")
        model, args, device = load_model()
        print(f"✓ 模型加载成功，使用设备: {device}")

        # 3. 创建分析结果目录
        result_dir = '../results/molecule_feature_importance/single_molecule_custom'
        os.makedirs(result_dir, exist_ok=True)
        
        # 4. 分析单个分子的特征重要性
        print(f"\n步骤3: 分析分子 {smiles} 的特征重要性...")
        atom_importance, feature_matrix = analyze_single_molecule_importance(
            smiles, model, data_dict, device, 
            save_path=f'{result_dir}/custom_molecule_analysis.png'
        )
        
        if atom_importance is not None:
            print(f"\n✓ 分析完成!")
            print(f"  - 原子数量: {len(atom_importance)}")
            print(f"  - 原子重要性分数范围: [{np.min(atom_importance):.6f}, {np.max(atom_importance):.6f}]")
            print(f"  - 平均原子重要性: {np.mean(atom_importance):.6f}")
            if feature_matrix is not None:
                print(f"  - 特征矩阵形状: {feature_matrix.shape}")
            
            # 保存数值结果
            np.save(f'{result_dir}/atom_importance.npy', atom_importance)
            if feature_matrix is not None:
                np.save(f'{result_dir}/feature_matrix.npy', feature_matrix)
            print(f"  - 结果保存在: {result_dir}")
            
            return atom_importance, feature_matrix
        else:
            print(f"✗ 分析失败!")
            return None, None
            
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """主函数"""
    print("单分子特征重要性分析使用示例")
    print("=" * 60)
    
    # 示例SMILES列表
    example_smiles_list = [
        "CC(=O)Nc1ccc(O)cc1",  # 对乙酰氨基酚
        "CC(C)C1CCC(C)CC1",    # 一种环状化合物
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因
    ]
    
    # 分析每个示例分子
    for i, smiles in enumerate(example_smiles_list, 1):
        print(f"\n{'='*20} 示例 {i} {'='*20}")
        analyze_custom_molecule(smiles)
    
    print("\n" + "=" * 60)
    print("🎉 所有示例分析完成!")


if __name__ == "__main__":
    main()