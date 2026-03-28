"""
Reusable baseline setup for lens finding task.
Used by both baseline and focal loss notebooks.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.insert(0, '../Lensing_DomainAdaptation')
from dataset import Len

class LensDataset(Dataset):
    """
    Custom dataset wrapper that handles image preprocessing for lens finding.
    - Loads images from .npy files
    - Converts multi-channel (3, 64, 64) to single-channel via channel averaging
    - Normalizes to [0, 1] range
    - Returns tensors in PyTorch format: [C, H, W]
    """
    def __init__(self, data_paths, augmentations=None):
        self.base_dataset = Len(data_paths, augmentations)
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        path = self.base_dataset.data[idx][0]
        label = float(self.base_dataset.data[idx][1])
        
        image = np.load(path)
        
        if image.ndim == 3:
            image = np.mean(image, axis=0)
        
        img_min = np.min(image)
        img_max = np.max(image)
        image = (image - img_min) / (img_max - img_min) if img_max > img_min else image
        
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        
        if self.augmentations is not None:
            transformed = self.augmentations(image=image)
            image = transformed['image']
        
        image_tensor = torch.tensor(image, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1)
        
        return image_tensor, torch.tensor(label, dtype=torch.long)


def create_dataloaders(X_train, X_test, train_aug=None, val_aug=None,
                       batch_size_train=64, batch_size_test=100):
    """
    Create DataLoaders for train and test sets using LensDataset.
    """
    train_data = LensDataset(X_train, train_aug)
    test_data = LensDataset(X_test, val_aug)
    
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        num_workers=0,
        batch_size=batch_size_train,
        drop_last=True,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_data,
        shuffle=False,
        num_workers=0,
        batch_size=batch_size_test,
        drop_last=False
    )
    
    return train_loader, test_loader