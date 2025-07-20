import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd


class BrainTumorTrainer:

    def __init__(self, model_class, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_class = model_class
        self.device = device
        self.model = None
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []
        }

    def evaluate_model(self, model, test_loader):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                outputs = model(inputs).squeeze()
                probs = torch.sigmoid(outputs)

                all_preds.extend((probs > 0.5).float().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs)
        }

        self._print_evaluation_results(metrics)
        return metrics

    def _print_evaluation_results(self, metrics):
        """Print formatted evaluation results."""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

    def plot_training_history(self, model=None):
        if model is None:
            model = self.model

        if model is None:
            print("No model available for plotting training history.")
            return

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(model.train_losses, label='Train Loss', color='blue')
        plt.plot(model.val_losses, label='Val Loss', color='red')
        plt.title('Training History - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(model.train_accs, label='Train Acc', color='blue')
        plt.plot(model.val_accs, label='Val Acc', color='red')
        plt.title('Training History - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def train_model(self, dataloaders, params=None, class_weights=None):
        # Default parameters
        default_params = {
            'dropout_rate': 0.07,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'epochs': 100,
            'patience': 15
        }

        if params is None:
            params = default_params
        else:
            # Merge with defaults
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value

        self._print_training_header(params)

        # Initialize model
        print("\nInitializing model...")
        self.model = self.model_class(dropout_rate=params['dropout_rate'])

        # Train the model
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

        # Evaluate the model
        print("\nEvaluating trained model...")
        metrics = self.evaluate_model(self.model, dataloaders['test'])

        return self.model, metrics

    def _print_training_header(self, params):
        """Print formatted training information."""
        print("Starting Custom CNN Training")
        print("Focus: CNN Architecture & Hyperparameter Optimization")
        print("=" * 60)
        print(f"\nUsing parameters: {params}")

    def run_complete_training_pipeline(self, dataloaders, params=None, class_weights=None,
                                       plot_history=True):
        # Train the model
        model, metrics = self.train_model(dataloaders, params, class_weights)

        # Plot training history if requested
        if plot_history:
            print("\nPlotting training history...")
            self.plot_training_history(model)

        return model, metrics

    def save_model(self, filepath, model=None):
        if model is None:
            model = self.model

        if model is None:
            print("No model available to save.")
            return

        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath, model_params=None):
        if model_params is None:
            model_params = {'dropout_rate': 0.07}

        self.model = self.model_class(**model_params)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")
        return self.model


# Example usage:
"""
# Initialize trainer
trainer = BrainTumorTrainer(BrainTumorCNN, device='cuda')

# Define training parameters
params = {
    'dropout_rate': 0.07,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'epochs': 100,
    'patience': 15
}

# Run complete training pipeline
model, metrics = trainer.run_complete_training_pipeline(
    dataloaders=dataloaders,
    params=params,
    class_weights=class_weights,
    plot_history=True
)

# Save the trained model
trainer.save_model('best_brain_tumor_model.pth')

# Or run individual components
# model, metrics = trainer.train_model(dataloaders, params, class_weights)
# trainer.plot_training_history()
# trainer.evaluate_model(model, dataloaders['test'])
"""