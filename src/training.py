"""
Training logic for brain tumor detection models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from tqdm import tqdm
import copy
from pathlib import Path

from .config import TRAINING_CONFIG, MODELS_DIR
from .models import create_model


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class ModelTrainer:
    """Main training class for brain tumor detection models."""
    
    def __init__(self, model_name, num_classes=4, device='cuda'):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        
        # Create model
        self.model = create_model(model_name, num_classes)
        self.model.to(device)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def _setup_optimizer(self, learning_rate, optimizer_type='adam'):
        """Setup optimizer and scheduler."""
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, 
                                     momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', 
                                         factor=0.5, patience=5, verbose=True)
    
    def _train_epoch(self, train_loader):
        """Train model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, val_loader):
        """Validate model for one epoch."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train_feature_extraction(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """Train model with frozen feature layers."""
        print(f"Training {self.model_name} with feature extraction...")
        
        # Freeze feature layers
        self.model.freeze_feature_layers()
        self._setup_optimizer(learning_rate)
        
        early_stopping = EarlyStopping(patience=TRAINING_CONFIG['early_stopping_patience'])
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
            print()
            
            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        # Save model
        model_path = MODELS_DIR / f"{self.model_name}_feature_extraction.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        return best_acc
    
    def train_fine_tuning(self, train_loader, val_loader, epochs=50, learning_rate=0.0001):
        """Train model with fine-tuning (all layers unfrozen)."""
        print(f"Fine-tuning {self.model_name}...")
        
        # Unfreeze all layers
        self.model.unfreeze_feature_layers()
        self._setup_optimizer(learning_rate)
        
        early_stopping = EarlyStopping(patience=TRAINING_CONFIG['early_stopping_patience'])
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
            print()
            
            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        # Save model
        model_path = MODELS_DIR / f"{self.model_name}_fine_tuning.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        return best_acc
    
    def get_training_history(self):
        """Get training history."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
