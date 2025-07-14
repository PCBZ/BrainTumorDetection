"""
Configuration management for brain tumor detection project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'vgg16': {
        'name': 'VGG16',
        'pretrained': True,
        'num_classes': 4,  # Adjust based on your dataset
        'input_size': 224,
        'feature_extraction_lr': 0.001,
        'fine_tuning_lr': 0.0001,
    },
    'resnet50': {
        'name': 'ResNet50',
        'pretrained': True,
        'num_classes': 4,
        'input_size': 224,
        'feature_extraction_lr': 0.001,
        'fine_tuning_lr': 0.0001,
    },
    'efficientnet_b0': {
        'name': 'EfficientNet-B0',
        'pretrained': True,
        'num_classes': 4,
        'input_size': 224,
        'feature_extraction_lr': 0.001,
        'fine_tuning_lr': 0.0001,
    }
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'random_seed': 42,
    'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'brightness_range': (0.8, 1.2),
}

# Visualization configuration
VIS_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
}
