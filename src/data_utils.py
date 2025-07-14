"""
Data utilities for brain tumor detection project.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path

from .config import TRAINING_CONFIG, AUGMENTATION_CONFIG


class BrainTumorDataset(Dataset):
    """Custom dataset for brain tumor images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms(input_size=224, is_training=True):
    """Get data transformation pipeline."""
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(AUGMENTATION_CONFIG['rotation_range']),
            transforms.RandomHorizontalFlip(p=0.5 if AUGMENTATION_CONFIG['horizontal_flip'] else 0),
            transforms.ColorJitter(brightness=AUGMENTATION_CONFIG['brightness_range']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def load_data(data_dir, test_size=0.2, val_size=0.1, random_state=42):
    """Load and split data into train, validation, and test sets."""
    
    # This is a placeholder implementation
    # You'll need to adapt this based on your actual data structure
    
    image_paths = []
    labels = []
    
    # Example: assuming data_dir has subdirectories for each class
    data_path = Path(data_dir)
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_path / class_name
        for img_file in class_dir.glob('*.jpg'):  # Adjust file extension as needed
            image_paths.append(str(img_file))
            labels.append(class_idx)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=test_size + val_size, 
        random_state=random_state, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size + val_size), 
        random_state=random_state, stratify=y_temp
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names


def create_data_loaders(train_data, val_data, test_data, batch_size=32, input_size=224):
    """Create data loaders for training, validation, and testing."""
    
    train_transform = get_data_transforms(input_size, is_training=True)
    val_transform = get_data_transforms(input_size, is_training=False)
    
    train_dataset = BrainTumorDataset(train_data[0], train_data[1], train_transform)
    val_dataset = BrainTumorDataset(val_data[0], val_data[1], val_transform)
    test_dataset = BrainTumorDataset(test_data[0], test_data[1], val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def preprocess_image(image_path, input_size=224):
    """Preprocess a single image for inference."""
    
    transform = get_data_transforms(input_size, is_training=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def calculate_dataset_statistics(data_loader):
    """Calculate mean and std of dataset for normalization."""
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean, std
