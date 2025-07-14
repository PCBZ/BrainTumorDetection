"""
Main experiment script for brain tumor detection project.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from datetime import datetime

from .config import MODEL_CONFIG, TRAINING_CONFIG, REPORTS_DIR, FIGURES_DIR
from .data_utils import load_data, create_data_loaders
from .training import ModelTrainer
from .evaluation import ModelEvaluator, compare_models
from .visualization import save_all_visualizations


def run_single_experiment(model_name, data_dir, num_classes=4, device='cuda'):
    """Run experiment for a single model."""
    
    print(f"\\n{'='*50}")
    print(f"Starting experiment for {model_name}")
    print(f"{'='*50}")
    
    # Load data
    print("Loading data...")
    train_data, val_data, test_data, class_names = load_data(data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, 
        batch_size=TRAINING_CONFIG['batch_size'],
        input_size=MODEL_CONFIG[model_name]['input_size']
    )
    
    print(f"Data loaded successfully!")
    print(f"Training samples: {len(train_data[0])}")
    print(f"Validation samples: {len(val_data[0])}")
    print(f"Test samples: {len(test_data[0])}")
    print(f"Classes: {class_names}")
    
    # Initialize trainer
    trainer = ModelTrainer(model_name, num_classes, device)
    
    # Feature extraction training
    print("\\n--- Feature Extraction Training ---")
    fe_acc = trainer.train_feature_extraction(
        train_loader, val_loader,
        epochs=TRAINING_CONFIG['epochs'] // 2,
        learning_rate=MODEL_CONFIG[model_name]['feature_extraction_lr']
    )
    
    # Fine-tuning training
    print("\\n--- Fine-tuning Training ---")
    ft_acc = trainer.train_fine_tuning(
        train_loader, val_loader,
        epochs=TRAINING_CONFIG['epochs'] // 2,
        learning_rate=MODEL_CONFIG[model_name]['fine_tuning_lr']
    )
    
    # Get training history
    history = trainer.get_training_history()
    
    # Evaluation
    print("\\n--- Evaluation ---")
    
    # Evaluate feature extraction model
    fe_model_path = REPORTS_DIR.parent / "results" / "models" / f"{model_name}_feature_extraction.pth"
    fe_evaluator = ModelEvaluator(model_name, fe_model_path, num_classes, class_names, device)
    fe_metrics = fe_evaluator.evaluate(test_loader)
    fe_evaluator.save_results(fe_metrics, f"feature_extraction")
    
    # Evaluate fine-tuning model
    ft_model_path = REPORTS_DIR.parent / "results" / "models" / f"{model_name}_fine_tuning.pth"
    ft_evaluator = ModelEvaluator(model_name, ft_model_path, num_classes, class_names, device)
    ft_metrics = ft_evaluator.evaluate(test_loader)
    ft_evaluator.save_results(ft_metrics, f"fine_tuning")
    
    # Return results
    return {
        'feature_extraction': {
            'metrics': fe_metrics,
            'best_val_acc': fe_acc,
            'history': history
        },
        'fine_tuning': {
            'metrics': ft_metrics,
            'best_val_acc': ft_acc,
            'history': history
        }
    }


def run_all_experiments(data_dir, models=['vgg16', 'resnet50', 'efficientnet_b0'], 
                       num_classes=4, device='cuda'):
    """Run experiments for all specified models."""
    
    print("Starting comprehensive brain tumor detection experiments...")
    print(f"Models to evaluate: {models}")
    print(f"Device: {device}")
    
    all_results = {}
    training_histories = {}
    
    for model_name in models:
        try:
            results = run_single_experiment(model_name, data_dir, num_classes, device)
            all_results[model_name] = results
            training_histories[model_name] = results['fine_tuning']['history']
            
            print(f"✓ {model_name} completed successfully")
            
        except Exception as e:
            print(f"✗ {model_name} failed: {str(e)}")
            continue
    
    # Compare models
    print("\\n--- Model Comparison ---")
    
    # Extract metrics for comparison
    comparison_results = {}
    for model_name, results in all_results.items():
        comparison_results[f"{model_name}_fe"] = results['feature_extraction']['metrics']
        comparison_results[f"{model_name}_ft"] = results['fine_tuning']['metrics']
    
    # Generate comparison
    comparison_df = compare_models(comparison_results)
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save experiment summary
    summary = {
        'timestamp': timestamp,
        'models_evaluated': models,
        'training_config': TRAINING_CONFIG,
        'model_configs': {model: MODEL_CONFIG[model] for model in models},
        'results': all_results,
        'comparison': comparison_df.to_dict()
    }
    
    summary_path = REPORTS_DIR / f"experiment_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        # Handle numpy arrays for JSON serialization
        summary_json = json.dumps(summary, indent=2, default=str)
        f.write(summary_json)
    
    # Save comparison CSV
    comparison_path = REPORTS_DIR / f"model_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    # Generate all visualizations
    save_all_visualizations(comparison_results, training_histories)
    
    print(f"\\n{'='*50}")
    print("All experiments completed!")
    print(f"Results saved to: {REPORTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"{'='*50}")
    
    return all_results, comparison_df


def main():
    """Main function to run experiments."""
    
    parser = argparse.ArgumentParser(description='Brain Tumor Detection Experiments')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the data directory')
    parser.add_argument('--models', nargs='+', 
                       default=['vgg16', 'resnet50', 'efficientnet_b0'],
                       help='Models to evaluate')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--single_model', type=str, default=None,
                       help='Run experiment for a single model')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Run experiments
    if args.single_model:
        results = run_single_experiment(args.single_model, args.data_dir, 
                                      args.num_classes, device)
        print(f"Single model experiment completed: {args.single_model}")
    else:
        results, comparison = run_all_experiments(args.data_dir, args.models, 
                                                args.num_classes, device)
        print("All experiments completed successfully!")
        print("\\nTop performing models:")
        print(comparison.sort_values('Accuracy', ascending=False).head())


if __name__ == "__main__":
    main()
