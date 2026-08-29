import os
import hashlib
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
# Note: processed parameter
def generate_smiles_hash(smiles):
    """ SHA256 """
    return hashlib.sha256(smiles.encode()).hexdigest()[:32] + ".png"  # 32


def batch_generate_images(input_csv, output_dir="smiles_images"):
    """ """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    processed = {}  # SMILES: ( Value,  )
    valid_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row.iloc[3]  # SMILES 3

        # SMILES
        if smiles in processed:
            valid_records.append({**row, "image_path": processed[smiles][1]})
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                raise ValueError("Invalid SMILES ")

            # Note: processed parameter
            file_hash = generate_smiles_hash(smiles)
            img_path = os.path.join(output_dir, file_hash)

            # Note: processed parameter
            if not os.path.exists(img_path):
                img = Draw.MolToImage(mol, size=(256, 256))
                img.save(img_path)

            # Note: processed parameter
            processed[smiles] = (file_hash, img_path)
            valid_records.append({**row, "image_path": img_path})

        except Exception as e:
            print(f"  {idx}  : {str(e)}")

    # Note: processed parameter
    clean_df = pd.DataFrame(valid_records)
    clean_df.to_csv("./clean_dataset.csv", index=False)
    return clean_df


def validate_uniqueness(clean_df):
    """ """
    hash_count = clean_df.groupby('image_path').size()
    duplicates = hash_count[hash_count > 1]

    if not duplicates.empty:
        print(f"  {len(duplicates)}  ")
    else:
        print(" SMILES ")


from torch.utils.data import Dataset
from PIL import Image


class CachedImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform or self.default_transform()

        # Note: processed parameter
        self.features = torch.tensor(
            self.df.iloc[:, 4:23].values, dtype=torch.float32
        )
        self.labels = torch.tensor(
            self.df.iloc[:, 23].values, dtype=torch.float32
        )
        self.image_paths = self.df["image_path"].tolist()

        # 1000
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
        # Note: processed parameter
        img_path = self.image_paths[idx]
        if img_path not in self.cache:
            img = Image.open(img_path).convert("RGB")
            self.cache[img_path] = img
            if len(self.cache) > 1000:  # LRU
                self.cache.pop(next(iter(self.cache)))

        img = self.transform(self.cache[img_path])
        return self.features[idx], img, self.labels[idx]


# 1.
clean_data = batch_generate_images(" .csv")

# 2.
validate_uniqueness(clean_data)

# 3.
dataset = CachedImageDataset("./clean_dataset.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)