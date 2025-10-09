from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image  # 用于显示图像
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
# 定义测试的SMILES字符串
test_smiles = [
    "CCO",  # 乙醇
    "c1ccccc1",  # 苯
    "CN1C=NC2=C1N=CN2C3=CC=CO3",  # 咖啡因
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
    "O=C1CCCC2=CC=CC=C12",  # 萘普生
    "CCNC1=NC(=NC(=N1)Cl)NC(C)C"
]

# 创建图像转换管道（与Dataset类中使用的相同）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 用于反转归一化的函数，以便可视化
def denormalize(tensor):
    """将归一化的图像张量转换回可显示的PIL图像"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean  # 反转归一化
    tensor = tensor.clamp(0, 1)  # 裁剪到有效范围
    tensor = tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
    return Image.fromarray((tensor * 255).astype(np.uint8))


# 测试每个SMILES
for i, smile in enumerate(test_smiles):
    try:
        print(f"\n处理分子 {i + 1}: {smile}")

        # 1. 将SMILES转换为分子对象
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print(f"  ✘ 无效的SMILES: {smile}")
            continue

        # 2. 生成原始图像
        img = Draw.MolToImage(mol, size=(224, 224))

        # 3. 转换为张量并应用预处理
        img_tensor = transform(img.convert('RGB'))

        # 4. 反转预处理进行可视化
        denormalized_img = denormalize(img_tensor)

        # 5. 显示结果
        plt.figure(figsize=(12, 4))

        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f"原始图像\n{smile}")
        plt.axis('off')

        # 处理后的图像
        plt.subplot(1, 3, 2)
        plt.imshow(denormalized_img)
        plt.title("处理后图像")
        plt.axis('off')

        # 分子结构
        plt.subplot(1, 3, 3)
        plt.imshow(Draw.MolToImage(mol, size=(200, 200), kekulize=True))
        plt.title("分子结构")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # 打印张量信息
        print(f"  ✔ 图像张量形状: {img_tensor.shape}")
        print(f"    最小值: {img_tensor.min().item():.4f}, 最大值: {img_tensor.max().item():.4f}")
        print(
            f"    均值: [{img_tensor[0].mean().item():.4f}, {img_tensor[1].mean().item():.4f}, {img_tensor[2].mean().item():.4f}]")

    except Exception as e:
        print(f"  处理 {smile} 时出错: {str(e)}")

print("\n测试完成！")