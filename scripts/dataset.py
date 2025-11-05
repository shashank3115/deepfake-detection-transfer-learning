import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import json

class DeepfakeDataset(Dataset):
    def __init__(self, json_path, img_root, transform=None):
        with open(json_path, "r") as f:
            self.meta = json.load(f)
        self.img_root = img_root
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        rec = self.meta[idx]
        label = rec["label"]  # 1 for real, 0 for fake
        # use index to find the right folder (optional, we can randomize later)
        folder = "real_cifake_images" if label == 1 else "fake_cifake_images"
        img_folder = os.path.join(self.img_root, folder)

        # pick image filename by index
        img_filename = f"{rec['index']}.png"  # adjust if JPG, JPEG, etc.
        img_path = os.path.join(img_folder, img_filename)

        # read image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)
