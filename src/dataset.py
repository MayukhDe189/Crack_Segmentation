import os
import glob
import json
import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader

class CrackDataset(Dataset):
    """
    CrackForest Dataset (CFD) loader.
    Loads pavement photographs (JPEG) and ground truth masks (.mat),
    resizes both to (256, 256), binarizes the mask (crack=1, bg=0),
    and applies standard normalization / data augmentation.
    Caches items in RAM after first load for fast training.
    """
    def __init__(self, sample_ids, img_dir, gt_dir, target_size=(256, 256), augment=False, cache_in_ram=True):
        self.sample_ids = sample_ids
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.target_size = target_size
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.cache = {}

        if self.cache_in_ram:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            for sid in self.sample_ids:
                img_path = os.path.join(self.img_dir, f"{sid}.jpg")
                gt_path = os.path.join(self.gt_dir, f"{sid}.mat")

                img = Image.open(img_path).convert("RGB")
                img = img.resize(self.target_size, Image.BILINEAR)
                img_np = np.array(img, dtype=np.float32) / 255.0

                mat_data = scipy.io.loadmat(gt_path)
                raw_seg = mat_data["groundTruth"][0, 0]["Segmentation"]
                mask_binary = (raw_seg == 2).astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_binary).resize(self.target_size, Image.NEAREST)
                mask_np = (np.array(mask_img, dtype=np.float32) > 127).astype(np.float32)

                img_norm = (img_np - mean) / std
                self.cache[sid] = (img_norm, mask_np)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        if self.cache_in_ram and sample_id in self.cache:
            img_norm, mask_np = self.cache[sample_id]
            img_norm = img_norm.copy()
            mask_np = mask_np.copy()
        else:
            img_path = os.path.join(self.img_dir, f"{sample_id}.jpg")
            gt_path = os.path.join(self.gt_dir, f"{sample_id}.mat")

            img = Image.open(img_path).convert("RGB")
            img = img.resize(self.target_size, Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0

            mat_data = scipy.io.loadmat(gt_path)
            raw_seg = mat_data["groundTruth"][0, 0]["Segmentation"]
            mask_binary = (raw_seg == 2).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_binary).resize(self.target_size, Image.NEAREST)
            mask_np = (np.array(mask_img, dtype=np.float32) > 127).astype(np.float32)

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_np - mean) / std

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img_norm = np.fliplr(img_norm).copy()
                mask_np = np.fliplr(mask_np).copy()
            if np.random.rand() > 0.5:
                img_norm = np.flipud(img_norm).copy()
                mask_np = np.flipud(mask_np).copy()

        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        return img_tensor, mask_tensor, sample_id


def get_data_splits(total_samples=118, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Creates deterministic train, val, and test splits.
    For CFD: 118 samples:
      Train: 82 samples (69.5%)
      Val:   18 samples (15.25%)
      Test:  18 samples (15.25%)
    """
    np.random.seed(seed)
    all_ids = [f"{i:03d}" for i in range(1, total_samples + 1)]
    shuffled_ids = np.random.permutation(all_ids).tolist()

    n_train = int(len(all_ids) * train_ratio)
    n_val = int(len(all_ids) * val_ratio)
    
    train_ids = sorted(shuffled_ids[:n_train])
    val_ids = sorted(shuffled_ids[n_train:n_train + n_val])
    test_ids = sorted(shuffled_ids[n_train + n_val:])

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }
