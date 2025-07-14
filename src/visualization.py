"""
Visualization utilities for brain tumor detection project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json

from .config import FIGURES_DIR, VIS_CONFIG


# Set matplotlib style
plt.style.use(VIS_CONFIG['style'])
sns.set_palette(VIS_CONFIG['color_palette'])


def plot_data_distribution(class_counts, class_names, save_path=None):
    """Plot data distribution across classes."""
    
    plt.figure(figsize=VIS_CONFIG['figure_size'])
    
    # Create bar plot
    bars = plt.bar(class_names, class_counts, color=sns.color_palette("viridis", len(class_names)))
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{count}', ha='center', va='bottom')
    
    plt.title('Data Distribution Across Classes', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training history (loss and accuracy curves)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss
    axes[0].plot(history['train_losses'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_losses'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['train_accuracies'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracies'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(model_results, metrics=['accuracy', 'precision', 'recall', 'f1'], save_path=None):
    """Plot comparison of multiple models."""
    
    # Prepare data
    models = list(model_results.keys())
    n_models = len(models)
    n_metrics = len(metrics)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = sns.color_palette("viridis", n_models)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [model_results[model][metric] for model in models]
        
        bars = ax.bar(models, values, color=colors)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(max(models, key=len)) > 8:
            ax.set_xticklabels(models, rotation=45, ha='right')
        
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()


def plot_performance_radar(model_results, save_path=None):
    """Create radar chart for model performance comparison."""
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(model_results.keys())
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = sns.color_palette("viridis", len(models))
    
    for i, model in enumerate(models):
        # Get values for this model
        values = [model_results[model][metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.capitalize() for metric in metrics])
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Add grid
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()


def plot_sample_predictions(images, true_labels, predictions, class_names, 
                          probabilities=None, n_samples=8, save_path=None):
    """Plot sample predictions with images."""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(n_samples, len(images))):
        ax = axes[i]
        
        # Display image
        if images[i].shape[0] == 3:  # If channels first
            img = np.transpose(images[i], (1, 2, 0))
        else:
            img = images[i]
        
        # Normalize for display
        img = (img - img.min()) / (img.max() - img.min())
        
        ax.imshow(img)
        ax.axis('off')
        
        # Create title
        true_class = class_names[true_labels[i]]
        pred_class = class_names[predictions[i]]
        
        color = 'green' if true_labels[i] == predictions[i] else 'red'
        
        title = f'True: {true_class}\\nPred: {pred_class}'
        
        if probabilities is not None:
            confidence = probabilities[i][predictions[i]]
            title += f'\\nConf: {confidence:.2f}'
        
        ax.set_title(title, color=color, fontsize=10)
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()


def create_experiment_summary_plot(experiment_results, save_path=None):
    """Create comprehensive experiment summary visualization."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Define grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Model comparison (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    models = list(experiment_results.keys())
    accuracies = [experiment_results[model]['accuracy'] for model in models]
    bars = ax1.bar(models, accuracies, color=sns.color_palette("viridis", len(models)))
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. Training curves (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    # This would need training history data
    ax2.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.5, 'Training curves would go here\\n(requires training history)', 
             ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Confusion matrices (bottom row)
    for i, model in enumerate(models[:3]):  # Show first 3 models
        ax = fig.add_subplot(gs[1:, i])
        cm = experiment_results[model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model} Confusion Matrix', fontsize=12)
    
    # 4. Summary statistics (bottom right)
    ax4 = fig.add_subplot(gs[1:, 3])
    summary_data = []
    for model in models:
        summary_data.append([
            model,
            f"{experiment_results[model]['accuracy']:.3f}",
            f"{experiment_results[model]['precision']:.3f}",
            f"{experiment_results[model]['recall']:.3f}",
            f"{experiment_results[model]['f1']:.3f}"
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.axis('off')
    ax4.set_title('Performance Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('Brain Tumor Detection - Experiment Summary', fontsize=18, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()


def save_all_visualizations(experiment_results, training_histories=None):
    """Save all visualizations to the figures directory."""
    
    # Model comparison
    comparison_path = FIGURES_DIR / "model_comparison.png"
    plot_model_comparison(experiment_results, save_path=comparison_path)
    
    # Performance radar
    radar_path = FIGURES_DIR / "performance_radar.png"
    plot_performance_radar(experiment_results, save_path=radar_path)
    
    # Training curves (if available)
    if training_histories:
        for model_name, history in training_histories.items():
            history_path = FIGURES_DIR / f"training_curves_{model_name}.png"
            plot_training_history(history, save_path=history_path)
    
    # Experiment summary
    summary_path = FIGURES_DIR / "experiment_summary.png"
    create_experiment_summary_plot(experiment_results, save_path=summary_path)
    
    print(f"All visualizations saved to {FIGURES_DIR}")
