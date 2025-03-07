import os
import hashlib
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
# 没用到
def generate_smiles_hash(smiles):
    """生成唯一的SHA256哈希文件名"""
    return hashlib.sha256(smiles.encode()).hexdigest()[:32] + ".png"  # 取前32字符缩短文件名


def batch_generate_images(input_csv, output_dir="smiles_images"):
    """批量生成图像并自动去重"""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    processed = {}  # 记录已处理的SMILES: (哈希值, 图像路径)
    valid_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row.iloc[3]  # 假设SMILES在第3列

        # 跳过已处理的SMILES
        if smiles in processed:
            valid_records.append({**row, "image_path": processed[smiles][1]})
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                raise ValueError("无效的SMILES结构")

            # 生成唯一文件名
            file_hash = generate_smiles_hash(smiles)
            img_path = os.path.join(output_dir, file_hash)

            # 仅当文件不存在时才生成
            if not os.path.exists(img_path):
                img = Draw.MolToImage(mol, size=(256, 256))
                img.save(img_path)

            # 更新记录
            processed[smiles] = (file_hash, img_path)
            valid_records.append({**row, "image_path": img_path})

        except Exception as e:
            print(f"行 {idx} 错误: {str(e)}")

    # 保存清洗后的数据集
    clean_df = pd.DataFrame(valid_records)
    clean_df.to_csv("./clean_dataset.csv", index=False)
    return clean_df


def validate_uniqueness(clean_df):
    """验证哈希唯一性"""
    hash_count = clean_df.groupby('image_path').size()
    duplicates = hash_count[hash_count > 1]

    if not duplicates.empty:
        print(f"发现 {len(duplicates)} 个重复图像路径")
    else:
        print("所有SMILES均对应唯一图像文件")


from torch.utils.data import Dataset
from PIL import Image


class CachedImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform or self.default_transform()

        # 预加载所有表格数据到内存
        self.features = torch.tensor(
            self.df.iloc[:, 4:23].values, dtype=torch.float32
        )
        self.labels = torch.tensor(
            self.df.iloc[:, 23].values, dtype=torch.float32
        )
        self.image_paths = self.df["image_path"].tolist()

        # 建立内存缓存（最多缓存1000张图像）
        self.cache = {}

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 从缓存获取或加载图像
        img_path = self.image_paths[idx]
        if img_path not in self.cache:
            img = Image.open(img_path).convert("RGB")
            self.cache[img_path] = img
            if len(self.cache) > 1000:  # LRU策略
                self.cache.pop(next(iter(self.cache)))

        img = self.transform(self.cache[img_path])
        return self.features[idx], img, self.labels[idx]


# 1. 生成去重图像
clean_data = batch_generate_images("原始数据.csv")

# 2. 验证数据
validate_uniqueness(clean_data)

# 3. 创建数据加载器
dataset = CachedImageDataset("./clean_dataset.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)