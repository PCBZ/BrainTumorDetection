"""

"""

import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import kagglehub

class DataPreprocessor:
    @staticmethod
    def organize_data(source_path, target_path='/content/brain_tumor_split', test_size=0.2):
        """
        Split data into train and test sets.
        """
        all_images, all_labels = [], []

        for class_name in ['yes', 'no']:
            class_path = os.path.join(source_path, class_name)
            images = os.listdir(class_path)
            for image in images:
                all_images.append(os.path.join(class_path, image))
                all_labels.append(class_name)

        # Split test set
        X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=test_size, stratify=all_labels, random_state=42)

        splits = {
            'train': (X_train, y_train),
            'test': (X_test, y_test)
        }

        for split, (images, labels) in splits.items():
            for class_name in ['yes', 'no']:
                class_path = os.path.join(target_path, split, class_name)
                os.makedirs(class_path, exist_ok=True)

            for image, label in zip(images, labels):
                shutil.copy(image, os.path.join(target_path, split, label))

        return target_path

class DataLoaderHelper:
    """
    Helper class to load data using PyTorch DataLoader.
    """
    @staticmethod
    def load_data(data_dir, image_size=(224, 224), batch_size=16, num_workers=2):
        """
        Load data from the specified directory.
        """
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
        
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
            for x in ['train', 'test']
        }

        dataloaders = {
            x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
            for x in ['train', 'test']
        }

        train_labels = [label for _, label in image_datasets['train']]
        class_counts = torch.bincount(torch.tensor(train_labels))
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * 2  # Normalize

        return dataloaders, class_weights, image_datasets

class DataUtils:
    @staticmethod
    def load_brain_tumor_data_pipeline(dataset_url='navoneel/brain-mri-images-for-brain-tumor-detection',
                                        target_path='/content/brain_tumor_data'):
        """
        Load brain tumor dataset from a URL and organize it into train/test splits.
        """
        # Download dataset
        path = kagglehub.dataset_download(dataset_url)
        
        # Organize data
        target_path = DataPreprocessor.organize_data(path, target_path)

        # Load data using DataLoader
        dataloaders, class_weights, image_datasets = DataLoaderHelper.load_data(target_path)

        return dataloaders, class_weights, image_datasets

if __name__ == "__main__":
    # Example usage
    dataloaders, class_weights, image_datasets = DataUtils.load_brain_tumor_data_pipeline(target_path='.data/brain_tumor_split')