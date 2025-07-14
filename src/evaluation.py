"""
Evaluation utilities for brain tumor detection models.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from .config import FIGURES_DIR, REPORTS_DIR
from .models import create_model


class ModelEvaluator:
    """Evaluation class for brain tumor detection models."""
    
    def __init__(self, model_name, model_path, num_classes=4, class_names=None, device='cuda'):
        self.model_name = model_name
        self.model_path = model_path
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.device = device
        
        # Load model
        self.model = create_model(model_name, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Results storage
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
    
    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        self.model.eval()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Get predictions and probabilities
                probs = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                self.predictions.extend(predicted.cpu().numpy())
                self.true_labels.extend(target.cpu().numpy())
                self.probabilities.extend(probs.cpu().numpy())
        
        return self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.true_labels, self.predictions)
        metrics['precision'] = precision_score(self.true_labels, self.predictions, average='weighted')
        metrics['recall'] = recall_score(self.true_labels, self.predictions, average='weighted')
        metrics['f1'] = f1_score(self.true_labels, self.predictions, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(self.true_labels, self.predictions, average=None)
        recall_per_class = recall_score(self.true_labels, self.predictions, average=None)
        f1_per_class = f1_score(self.true_labels, self.predictions, average=None)
        
        metrics['per_class'] = {
            'precision': dict(zip(self.class_names, precision_per_class)),
            'recall': dict(zip(self.class_names, recall_per_class)),
            'f1': dict(zip(self.class_names, f1_per_class))
        }
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(self.true_labels, self.predictions)
        
        # ROC-AUC for multiclass
        if self.num_classes > 2:
            y_bin = label_binarize(self.true_labels, classes=range(self.num_classes))
            metrics['roc_auc'] = roc_auc_score(y_bin, self.probabilities, multi_class='ovr')
        else:
            metrics['roc_auc'] = roc_auc_score(self.true_labels, 
                                             np.array(self.probabilities)[:, 1])
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix."""
        if not self.predictions:
            raise ValueError("No predictions available. Run evaluate() first.")
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, save_path=None):
        """Plot ROC curves for each class."""
        if not self.probabilities:
            raise ValueError("No probabilities available. Run evaluate() first.")
        
        y_bin = label_binarize(self.true_labels, classes=range(self.num_classes))
        
        plt.figure(figsize=(12, 8))
        
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], np.array(self.probabilities)[:, i])
            auc = roc_auc_score(y_bin[:, i], np.array(self.probabilities)[:, i])
            
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {self.model_name}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_classification_report(self, save_path=None):
        """Generate detailed classification report."""
        if not self.predictions:
            raise ValueError("No predictions available. Run evaluate() first.")
        
        report = classification_report(self.true_labels, self.predictions, 
                                     target_names=self.class_names, output_dict=True)
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def save_results(self, metrics, file_prefix="evaluation"):
        """Save evaluation results to files."""
        # Save metrics to JSON
        metrics_path = REPORTS_DIR / f"{file_prefix}_{self.model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_copy = metrics.copy()
            metrics_copy['confusion_matrix'] = metrics_copy['confusion_matrix'].tolist()
            json.dump(metrics_copy, f, indent=2)
        
        # Save confusion matrix plot
        cm_path = FIGURES_DIR / f"{file_prefix}_{self.model_name}_confusion_matrix.png"
        self.plot_confusion_matrix(cm_path)
        
        # Save ROC curves
        roc_path = FIGURES_DIR / f"{file_prefix}_{self.model_name}_roc_curves.png"
        self.plot_roc_curves(roc_path)
        
        # Save classification report
        report_path = REPORTS_DIR / f"{file_prefix}_{self.model_name}_classification_report.json"
        self.generate_classification_report(report_path)
        
        print(f"Results saved to {REPORTS_DIR} and {FIGURES_DIR}")
        
        return metrics_path, cm_path, roc_path, report_path


def compare_models(model_results, save_path=None):
    """Compare multiple models and create comparison plots."""
    
    # Extract metrics for comparison
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model_name, metrics in model_results.items():
        model_names.append(model_name)
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//2, i%2]
        bars = ax.bar(comparison_df['Model'], comparison_df[metric])
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return comparison_df
