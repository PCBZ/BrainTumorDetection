# API Documentation

## Brain Tumor Detection API

### Overview
This API provides functions for brain tumor detection using deep learning models including VGG16, ResNet50, and EfficientNet-B0.

### Core Modules

#### config.py
Configuration management for the project.

**Key Variables:**
- `MODEL_CONFIG`: Configuration for each model type
- `TRAINING_CONFIG`: Training parameters
- `AUGMENTATION_CONFIG`: Data augmentation settings
- `VIS_CONFIG`: Visualization settings

#### data_utils.py
Data loading and preprocessing utilities.

**Key Functions:**
- `load_data(data_dir, test_size=0.2, val_size=0.1, random_state=42)`
  - Loads and splits data into train/validation/test sets
  - Returns: train_data, val_data, test_data, class_names

- `create_data_loaders(train_data, val_data, test_data, batch_size=32, input_size=224)`
  - Creates PyTorch DataLoader objects
  - Returns: train_loader, val_loader, test_loader

- `preprocess_image(image_path, input_size=224)`
  - Preprocesses single image for inference
  - Returns: tensor ready for model input

#### models.py
Model definitions and factory functions.

**Key Classes:**
- `BrainTumorClassifier`: Base class for all models
- `VGG16Classifier`: VGG16-based classifier
- `ResNet50Classifier`: ResNet50-based classifier
- `EfficientNetB0Classifier`: EfficientNet-B0-based classifier

**Key Functions:**
- `create_model(model_name, num_classes=4, pretrained=True)`
  - Factory function to create model instances
  - Returns: model instance

- `count_parameters(model)`
  - Counts trainable parameters
  - Returns: number of parameters

#### training.py
Training utilities and procedures.

**Key Classes:**
- `ModelTrainer`: Main training class
- `EarlyStopping`: Early stopping utility

**Key Methods:**
- `train_feature_extraction(train_loader, val_loader, epochs=50, learning_rate=0.001)`
  - Trains with frozen feature layers
  - Returns: best validation accuracy

- `train_fine_tuning(train_loader, val_loader, epochs=50, learning_rate=0.0001)`
  - Trains with all layers unfrozen
  - Returns: best validation accuracy

#### evaluation.py
Model evaluation and metrics calculation.

**Key Classes:**
- `ModelEvaluator`: Evaluation utilities

**Key Methods:**
- `evaluate(test_loader)`
  - Evaluates model on test set
  - Returns: metrics dictionary

- `plot_confusion_matrix(save_path=None)`
  - Plots confusion matrix

- `plot_roc_curves(save_path=None)`
  - Plots ROC curves for each class

#### visualization.py
Visualization utilities for results and analysis.

**Key Functions:**
- `plot_data_distribution(class_counts, class_names, save_path=None)`
- `plot_training_history(history, save_path=None)`
- `plot_model_comparison(model_results, metrics, save_path=None)`
- `plot_performance_radar(model_results, save_path=None)`

#### experiment.py
Main experiment runner.

**Key Functions:**
- `run_single_experiment(model_name, data_dir, num_classes=4, device='cuda')`
  - Runs complete experiment for single model
  - Returns: results dictionary

- `run_all_experiments(data_dir, models, num_classes=4, device='cuda')`
  - Runs experiments for all specified models
  - Returns: all_results, comparison_df

### Usage Examples

#### Basic Model Training
```python
from src.training import ModelTrainer
from src.data_utils import load_data, create_data_loaders

# Load data
train_data, val_data, test_data, class_names = load_data('path/to/data')
train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)

# Train model
trainer = ModelTrainer('vgg16', num_classes=4, device='cuda')
trainer.train_feature_extraction(train_loader, val_loader)
trainer.train_fine_tuning(train_loader, val_loader)
```

#### Model Evaluation
```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator('vgg16', 'path/to/model.pth', num_classes=4, class_names=class_names)
metrics = evaluator.evaluate(test_loader)
evaluator.save_results(metrics)
```

#### Running Complete Experiments
```python
from src.experiment import run_all_experiments

results, comparison = run_all_experiments(
    data_dir='path/to/data',
    models=['vgg16', 'resnet50', 'efficientnet_b0'],
    num_classes=4,
    device='cuda'
)
```

### Command Line Interface

Run experiments from command line:
```bash
python -m src.experiment --data_dir /path/to/data --models vgg16 resnet50 efficientnet_b0
```

### Output Files

The system generates several output files:
- `results/models/`: Trained model weights (.pth files)
- `results/figures/`: Visualizations and plots (.png files)
- `results/reports/`: Metrics and analysis reports (.json, .csv files)

### Error Handling

All modules include comprehensive error handling for:
- Missing data files
- Invalid model names
- CUDA availability issues
- File I/O errors

### Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
