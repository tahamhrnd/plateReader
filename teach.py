import os
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms


class LicensePlateDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([int(c) for c in label], dtype=torch.long)
        return image, label
