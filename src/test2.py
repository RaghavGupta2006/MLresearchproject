import torch
from torch.utils.data import Dataset
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def generate_cropped_molecule_image(smiles, target_size=224, padding=0.07):
    """生成裁剪后居中显示的分子图像"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # 计算渲染尺寸（保留7%边界）
    render_size = int(target_size * (1 - 2 * padding))

    # 生成紧凑分子图像
    img = Draw.MolToImage(mol, size=(render_size, render_size), fitImage=True)

    # 创建目标画布（白色背景）
    canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))

    # 居中粘贴分子图像
    paste_position = (
        (target_size - render_size) // 2,
        (target_size - render_size) // 2
    )
    canvas.paste(img, paste_position)

    return canvas, mol


def visualize_molecule_examples(smiles_list, titles=None, cols=3):
    """可视化多个分子图像示例"""
    num_examples = len(smiles_list)
    rows = (num_examples + cols - 1) // cols

    plt.figure(figsize=(15, 5 * rows))

    for i, smile in enumerate(smiles_list):
        img, mol = generate_cropped_molecule_image(smile)
        if img is None:
            print(f"警告: 跳过无效的SMILES: {smile}")
            continue

        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.asarray(img))

        # 添加标题（显示分子信息或自定义标题）
        if titles and i < len(titles):
            title = titles[i]
        else:
            title = f"{smile}\n{mol.GetNumAtoms()} atoms"

        plt.title(title, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("molecule_examples.png", dpi=300, bbox_inches='tight')
    plt.show()
    return "molecule_examples.png"


# 改进后的图像数据集类
class OptimizedTabularImageDataset(Dataset):
    def __init__(self, X, smiles_list, y, image_transform=None):
        """
        Args:
            X (numpy.ndarray): 数值特征矩阵
            smiles_list (list): SMILES字符串列表
            y (numpy.ndarray): 标签数组
            image_transform (torchvision.transforms): 图像预处理变换
        """
        self.X = X.clone().detach().to(torch.float32) if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32)
        self.smiles_list = smiles_list
        self.y = y.clone().detach().to(torch.float32) if torch.is_tensor(y) else torch.tensor(y, dtype=torch.float32)

        # 优化后的默认转换流程
        self.image_transform = image_transform or transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 预渲染所有分子图像
        self.pre_render_images()

    def pre_render_images(self):
        """预处理渲染所有分子图像"""
        self.images = []
        for smi in self.smiles_list:
            img, _ = generate_cropped_molecule_image(smi)
            if img is None:
                raise ValueError(f"无效的SMILES: {smi}")
            self.images.append(img)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        table_feature = self.X[idx]
        label = self.y[idx]
        img = self.images[idx]

        # 应用变换
        if self.image_transform:
            img = self.image_transform(img)

        return table_feature, img, label


# 使用示例
if __name__ == "__main__":
    # 示例分子
    test_smiles = [
        "CCO",  # 乙醇
        "C1=CC=CC=C1",  # 苯
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
        "C1CCCCC1",  # 环己烷
        "O=C=O",  # 二氧化碳
        "C(Cl)(Cl)(Cl)Cl",  # 四氯化碳
        "N[C@@H](CC1=CC=CC=C1)C(O)=O",  # 苯丙氨酸
    ]

    # 可视化分子示例
    saved_path = visualize_molecule_examples(
        test_smiles,
        titles=[
            "乙醇 (Ethanol)",
            "苯 (Benzene)",
            "咖啡因 (Caffeine)",
            "阿司匹林 (Aspirin)",
            "环己烷 (Cyclohexane)",
            "二氧化碳 (CO₂)",
            "四氯化碳 (CCl₄)",
            "苯丙氨酸 (Phenylalanine)"
        ],
        cols=4
    )

    print(f"分子示例已保存至: {saved_path}")