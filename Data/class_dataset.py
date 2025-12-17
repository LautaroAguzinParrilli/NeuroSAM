import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, img_paths, ages, target_shape=(176, 208, 176)):
        self.img_paths = img_paths
        self.ages = ages
        self.target_shape = target_shape

    def __len__(self):
        return len(self.img_paths)

    def crop_center(self, img, target_shape):
        d, h, w = img.shape
        td, th, tw = target_shape
        start_d = (d - td) // 2
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        return img[start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]

    def __getitem__(self, idx):
        img = nib.load(self.img_paths[idx]).get_fdata()
        img = img.astype(np.float32)

        # Crop central
        img = self.crop_center(img, self.target_shape)

        # Normalización simple
        img = (img - img.mean()) / img.std()

        # (C, D, H, W)
        img = torch.tensor(img).unsqueeze(0)

        age = torch.tensor(self.ages[idx]).float()
        return img, age
