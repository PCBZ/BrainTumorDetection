import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd


class BrainTumorTrainer:
    """
    Compatible trainer class that works with external model classes
    Supports different approaches while maintaining flexibility
    """

    def __init__(self, model_class=None, device='cuda' if torch.cuda.is_available() else 'cpu', config=None):
        """
        Initialize trainer with model class or configuration
        
        Args:
            model_class: Model class to use (for approach1/2)
            device: Device to use for training
            config: Configuration dictionary with model_params (for approach3)
        """
        self.model_class = model_class
        self.device = device
        self.config = config
        self.model = None
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []
        }

    def create_model_from_config(self, config):
        """Create model from configuration - compatible with data hub approach"""
        model_params = config.get('model_params', {})
        
        # Import here to avoid circular imports
        from CNN_brain_tumor import create_model_from_config
        return create_model_from_config(config)

    def evaluate_model(self, model, test_loader):
        """Evaluate model performance with compatibility for different output formats"""
        model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                
                # Handle different model architectures
                if hasattr(model, 'num_classes') and model.num_classes == 2:
                    # Binary classification
                    if len(outputs.shape) > 1 and outputs.shape[1] == 1:
                        outputs = outputs.squeeze()
                    
                    labels = labels.float()
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    
                    all_probs.extend(probs.cpu().numpy())
                else:
                    # Multi-class classification or legacy binary
                    if len(outputs.shape) == 1:
                        # Legacy binary format
                        labels = labels.float()
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).float()
                        all_probs.extend(probs.cpu().numpy())
                    else:
                        # Multi-class format
                        labels = labels.long()
                        probs = torch.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)
                        preds = preds.float()
                        
                        # For multi-class, use max probability for AUC calculation
                        all_probs.extend(probs.max(dim=1)[0].cpu().numpy())

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0, average='binary' if len(set(all_labels)) == 2 else 'weighted'),
            'recall': recall_score(all_labels, all_preds, zero_division=0, average='binary' if len(set(all_labels)) == 2 else 'weighted'),
            'f1': f1_score(all_labels, all_preds, zero_division=0, average='binary' if len(set(all_labels)) == 2 else 'weighted'),
        }
        
        # Only calculate AUC for binary classification
        if len(set(all_labels)) == 2:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        else:
            metrics['auc'] = 'N/A (Multi-class)'

        self._print_evaluation_results(metrics)
        return metrics

    def _print_evaluation_results(self, metrics):
        """Print formatted evaluation results."""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        for metric, value in metrics.items():
            if isinstance(value, str):
                print(f"{metric.upper()}: {value}")
            else:
                print(f"{metric.upper()}: {value:.4f}")

    def plot_training_history(self, model=None):
        """Plot training history with compatibility for different model types"""
        if model is None:
            model = self.model

        if model is None:
            print("No model available for plotting training history.")
            return

        # Check if model has training history attributes
        train_losses = getattr(model, 'train_losses', [])
        val_losses = getattr(model, 'val_losses', [])
        train_accs = getattr(model, 'train_accs', [])
        val_accs = getattr(model, 'val_accs', [])

        if not train_losses:
            print("No training history available.")
            return

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.title('Training History - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc', color='blue')
        plt.plot(val_accs, label='Val Acc', color='red')
        plt.title('Training History - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def train_model(self, dataloaders, params=None, class_weights=None, config=None):
        """
        Train model with compatibility for different approaches
        
        Args:
            dataloaders: Dictionary with 'train' and 'test' dataloaders
            params: Training parameters (approach1/2)
            class_weights: Class weights for imbalanced datasets
            config: Configuration dictionary (approach3)
        """
        # Default parameters
        default_params = {
            'dropout_rate': 0.07,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'epochs': 100,
            'patience': 15
        }

        # Handle different parameter input methods
        if config is not None:
            # Approach 3: Use config
            self.config = config
            params = config.get('training_params', default_params)
            model_params = config.get('model_params', {})
        elif params is None:
            params = default_params
        else:
            # Merge with defaults
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value

        self._print_training_header(params)

        # Initialize model based on approach
        print("\nInitializing model...")
        if self.config is not None:
            # Approach 3: Create from config
            self.model = self.create_model_from_config(self.config)
        elif self.model_class is not None:
            # Approach 1/2: Use model class
            self.model = self.model_class(dropout_rate=params.get('dropout_rate', 0.07))
        else:
            raise ValueError("Either model_class or config must be provided")

        # Train the model if it has train_model method
        if hasattr(self.model, 'train_model'):
            print("Training model...")
            self.model.train_model(
                dataloaders['train'],
                dataloaders['test'],
                epochs=params['epochs'],
                lr=params['learning_rate'],
                optimizer_type=params['optimizer'],
                class_weights=class_weights,
                patience=params['patience']
            )
        else:
            print("Model does not have built-in training method. Using external training loop...")
            self._external_training_loop(dataloaders, params, class_weights)

        # Evaluate the model
        print("\nEvaluating trained model...")
        metrics = self.evaluate_model(self.model, dataloaders['test'])

        return self.model, metrics

    def _external_training_loop(self, dataloaders, params, class_weights):
        """External training loop for models without built-in training"""
        # This would implement a generic training loop
        # for models that don't have their own train_model method
        print("External training loop not implemented. Please use models with built-in training.")
        pass

    def _print_training_header(self, params):
        """Print formatted training information."""
        print("Starting Compatible CNN Training")
        print("Supports: approach1, approach2, approach3 with data hub")
        print("=" * 60)
        print(f"\nUsing parameters: {params}")

    def run_complete_training_pipeline(self, dataloaders, params=None, class_weights=None,
                                       plot_history=True, config=None):
        """
        Run complete training pipeline with compatibility for all approaches
        
        Args:
            dataloaders: Training and validation dataloaders
            params: Training parameters (approach1/2)
            class_weights: Class weights for imbalanced datasets
            plot_history: Whether to plot training history
            config: Configuration dictionary (approach3)
        """
        # Train the model
        model, metrics = self.train_model(dataloaders, params, class_weights, config)

        # Plot training history if requested
        if plot_history:
            print("\nPlotting training history...")
            self.plot_training_history(model)

        return model, metrics

    def save_model(self, filepath, model=None):
        """Save model state dict"""
        if model is None:
            model = self.model

        if model is None:
            print("No model available to save.")
            return

        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath, model_params=None, config=None):
        """Load model with compatibility for different approaches"""
        if config is not None:
            # Approach 3: Load from config
            self.config = config
            self.model = self.create_model_from_config(config)
        elif self.model_class is not None:
            # Approach 1/2: Use model class
            if model_params is None:
                model_params = {'dropout_rate': 0.07}
            self.model = self.model_class(**model_params)
        else:
            raise ValueError("Either model_class or config must be provided for loading")

        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")
        return self.model


# Example usage for different approaches:
"""
# Approach 1 & 2: Traditional model class approach
from CNN_brain_tumor import BrainTumorCNN

trainer = BrainTumorTrainer(model_class=BrainTumorCNN, device='cuda')

params = {
    'dropout_rate': 0.07,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'epochs': 100,
    'patience': 15
}

model, metrics = trainer.run_complete_training_pipeline(
    dataloaders=dataloaders,
    params=params,
    class_weights=class_weights,
    plot_history=True
)

# Approach 3: Configuration-based approach (data hub method)
config = {
    'model_type': 'SimpleCNN',  # or 'BrainTumorCNN'
    'model_params': {
        'num_classes': 2,
        'dropout_rate': 0.3
    },
    'training_params': {
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'epochs': 50,
        'patience': 10
    }
}

trainer = BrainTumorTrainer(config=config, device='cuda')

model, metrics = trainer.run_complete_training_pipeline(
    dataloaders=dataloaders,
    config=config,
    plot_history=True
)
"""
