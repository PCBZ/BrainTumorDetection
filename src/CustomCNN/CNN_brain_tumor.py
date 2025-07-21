import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3, **kwargs):
        """
        Compatible CNN model for brain tumor detection
        
        Args:
            num_classes (int): Number of output classes (2 for binary classification)
            dropout_rate (float): Dropout rate for regularization
            **kwargs: Additional parameters for compatibility
        """
        super(BrainTumorCNN, self).__init__()
        
        # Store configuration for compatibility
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate * 0.5),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate * 0.7),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classifier for binary or multi-class classification
        if num_classes == 2:
            # Binary classification - single output
            self.classifier = nn.Sequential(
                nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, 1)
            )
        else:
            # Multi-class classification
            self.classifier = nn.Sequential(
                nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )

        # Training attributes
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

    def forward(self, x):
        """Standard forward pass - required for compatibility"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_epoch(self, loader, criterion, optimizer, class_weights=None):
        self.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in tqdm(loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Handle different label formats for compatibility
            if self.num_classes == 2:
                labels = labels.float()
            else:
                labels = labels.long()

            optimizer.zero_grad()
            outputs = self(inputs)
            
            if self.num_classes == 2:
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                
                if class_weights is not None:
                    weights = class_weights[labels.long()]
                    loss = (loss * weights).mean()
                    
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                
                if class_weights is not None:
                    weights = class_weights[labels]
                    loss = (loss * weights).mean()
                    
                _, predicted = torch.max(outputs.data, 1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(loader), 100. * correct / total

    def validate_epoch(self, loader, criterion):
        self.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Handle different label formats for compatibility
                if self.num_classes == 2:
                    labels = labels.float()
                else:
                    labels = labels.long()
                
                outputs = self(inputs)
                
                if self.num_classes == 2:
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, labels)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                else:
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(torch.softmax(outputs, dim=1).cpu().numpy())

                total_loss += loss.item()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())

        return total_loss / len(loader), 100. * correct / total, all_preds, all_labels

    def train_model(self, train_loader, val_loader, epochs, lr, optimizer_type='adam',
                   class_weights=None, patience=15):
        self.to(device)
        
        # Choose appropriate loss function based on number of classes
        if self.num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        if class_weights is not None:
            class_weights = class_weights.to(device)

        optimizers = {
            'adam': optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4),
            'sgd': optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4),
            'rmsprop': optim.RMSprop(self.parameters(), lr=lr, weight_decay=1e-4)
        }
        optimizer = optimizers[optimizer_type]
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, class_weights)
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(val_loader, criterion)

            scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return val_preds, val_labels


# SimpleCNN - Basic model for approach3 compatibility
class SimpleCNN(nn.Module):
    """Simplified CNN model for approach3 data hub method"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3, **kwargs):
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        """Standard forward pass"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Factory function for model creation with config compatibility
def create_model_from_config(config):
    """
    Create model instance from configuration dictionary
    
    Args:
        config (dict): Configuration dictionary containing 'model_params'
        
    Returns:
        nn.Module: Initialized model instance
    """
    model_params = config.get('model_params', {})
    
    # Default parameters
    default_params = {
        'num_classes': 2,
        'dropout_rate': 0.3
    }
    
    # Merge with provided parameters
    for key, value in default_params.items():
        if key not in model_params:
            model_params[key] = value
    
    # Choose model type based on config or default to BrainTumorCNN
    model_type = config.get('model_type', 'BrainTumorCNN')
    
    if model_type == 'SimpleCNN':
        return SimpleCNN(**model_params)
    else:
        return BrainTumorCNN(**model_params)


# Example usage for different approaches:
"""
# Approach 1 & 2: Direct instantiation
config = {
    'model_params': {
        'num_classes': 2,
        'dropout_rate': 0.07
    }
}

model = BrainTumorCNN(**config['model_params'])

# Approach 3: Using SimpleCNN with data hub method
config = {
    'model_type': 'SimpleCNN',
    'model_params': {
        'num_classes': 2,
        'dropout_rate': 0.3
    }
}

model = create_model_from_config(config)

# All models have standard forward method and are nn.Module subclasses
output = model(input_tensor)
"""
