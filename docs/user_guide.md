# Brain Tumor Detection - User Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Running Experiments](#running-experiments)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- At least 8GB RAM
- 10GB free disk space

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd brain-tumor-detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
1. **Prepare your data** in the following structure:
   ```
   data/
   ├── class_1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── class_2/
   │   ├── image1.jpg
   │   └── ...
   └── ...
   ```

2. **Run a single model experiment:**
   ```bash
   python -m src.experiment --data_dir /path/to/data --single_model vgg16
   ```

3. **Run all models:**
   ```bash
   python -m src.experiment --data_dir /path/to/data
   ```

### Expected Output
After running experiments, you'll find:
- Trained models in `results/models/`
- Visualization plots in `results/figures/`
- Performance reports in `results/reports/`

## Data Preparation

### Supported Formats
- **Image formats**: JPG, PNG, JPEG
- **Image size**: Any size (will be resized to 224x224)
- **Color**: RGB images

### Data Structure
Your data should be organized in subdirectories where each subdirectory represents a class:

```
your_data/
├── glioma/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── meningioma/
│   ├── image_001.jpg
│   └── ...
├── no_tumor/
│   └── ...
└── pituitary/
    └── ...
```

### Data Quality Guidelines
- **Image quality**: Use high-resolution medical images
- **Consistency**: Ensure similar imaging conditions across classes
- **Balance**: Try to have similar number of samples per class
- **Validation**: Remove corrupted or mislabeled images

## Running Experiments

### Command Line Options
```bash
python -m src.experiment [OPTIONS]
```

**Required Arguments:**
- `--data_dir`: Path to your data directory

**Optional Arguments:**
- `--models`: List of models to train (default: all models)
- `--num_classes`: Number of classes (default: 4)
- `--device`: Device to use ('cuda' or 'cpu')
- `--single_model`: Train only one specific model

### Examples

#### Train All Models
```bash
python -m src.experiment --data_dir ./data --device cuda
```

#### Train Specific Models
```bash
python -m src.experiment --data_dir ./data --models vgg16 resnet50
```

#### Use CPU Only
```bash
python -m src.experiment --data_dir ./data --device cpu
```

#### Different Number of Classes
```bash
python -m src.experiment --data_dir ./data --num_classes 3
```

## Understanding Results

### Performance Metrics
The system reports several key metrics:

1. **Accuracy**: Overall classification accuracy
2. **Precision**: Precision per class and weighted average
3. **Recall**: Recall per class and weighted average
4. **F1-Score**: F1-score per class and weighted average
5. **ROC-AUC**: Area under the ROC curve

### Generated Files

#### Models (`results/models/`)
- `{model_name}_feature_extraction.pth`: Model trained with frozen features
- `{model_name}_fine_tuning.pth`: Model trained with fine-tuning

#### Figures (`results/figures/`)
- `model_comparison.png`: Bar chart comparing all models
- `performance_radar.png`: Radar chart showing performance across metrics
- `{model_name}_confusion_matrix.png`: Confusion matrix for each model
- `{model_name}_roc_curves.png`: ROC curves for each model
- `training_curves_{model_name}.png`: Training/validation curves

#### Reports (`results/reports/`)
- `experiment_summary_{timestamp}.json`: Complete experiment results
- `model_comparison_{timestamp}.csv`: Comparison table
- `{model_name}_metrics.json`: Detailed metrics for each model

### Interpreting Results

#### Confusion Matrix
- **Diagonal elements**: Correct predictions
- **Off-diagonal elements**: Misclassifications
- **Dark blue**: High values, **Light blue**: Low values

#### ROC Curves
- **Closer to top-left**: Better performance
- **AUC > 0.9**: Excellent performance
- **AUC > 0.8**: Good performance
- **AUC > 0.7**: Fair performance

#### Training Curves
- **Converging lines**: Good training
- **Diverging lines**: Potential overfitting
- **Fluctuating validation**: May need more regularization

## Advanced Usage

### Custom Configuration
Modify `src/config.py` to customize:
- Model architectures
- Training parameters
- Data augmentation settings
- Visualization preferences

### Using Individual Components

#### Data Loading
```python
from src.data_utils import load_data, create_data_loaders

# Load your data
train_data, val_data, test_data, class_names = load_data('path/to/data')
train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)
```

#### Model Training
```python
from src.training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer('vgg16', num_classes=4, device='cuda')

# Train with feature extraction
trainer.train_feature_extraction(train_loader, val_loader, epochs=50)

# Fine-tune the model
trainer.train_fine_tuning(train_loader, val_loader, epochs=50)
```

#### Model Evaluation
```python
from src.evaluation import ModelEvaluator

# Evaluate model
evaluator = ModelEvaluator('vgg16', 'path/to/model.pth', num_classes=4, class_names=class_names)
metrics = evaluator.evaluate(test_loader)

# Generate visualizations
evaluator.plot_confusion_matrix()
evaluator.plot_roc_curves()
```

### Hyperparameter Tuning
To modify hyperparameters, edit `src/config.py`:

```python
MODEL_CONFIG = {
    'vgg16': {
        'feature_extraction_lr': 0.001,  # Learning rate for feature extraction
        'fine_tuning_lr': 0.0001,        # Learning rate for fine-tuning
        # ... other parameters
    }
}

TRAINING_CONFIG = {
    'batch_size': 32,                    # Batch size
    'epochs': 100,                       # Number of epochs
    'early_stopping_patience': 10,       # Early stopping patience
    # ... other parameters
}
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`
**Solution**: 
- Reduce batch size in `src/config.py`
- Use smaller input image size
- Use CPU instead: `--device cpu`

#### 2. Module Import Errors
**Error**: `ModuleNotFoundError: No module named 'torch'`
**Solution**:
- Ensure virtual environment is activated
- Install requirements: `pip install -r requirements.txt`

#### 3. Data Loading Issues
**Error**: `FileNotFoundError` or empty datasets
**Solution**:
- Check data directory structure
- Ensure images are in correct format
- Verify file permissions

#### 4. Memory Issues During Training
**Error**: System runs out of RAM
**Solution**:
- Reduce batch size
- Use fewer workers in DataLoader
- Close other applications

#### 5. Poor Performance
**Issue**: Low accuracy or overfitting
**Solution**:
- Check data quality and balance
- Increase data augmentation
- Adjust learning rates
- Use early stopping

### Performance Optimization

#### For Better Speed:
1. Use GPU if available
2. Increase batch size (if memory allows)
3. Use mixed precision training
4. Reduce image resolution if appropriate

#### For Better Accuracy:
1. Use more training epochs
2. Implement cross-validation
3. Use ensemble methods
4. Collect more training data
5. Apply better data augmentation

### Getting Help

If you encounter issues:
1. Check the error message carefully
2. Look for similar issues in the documentation
3. Verify your data format and structure
4. Check system requirements
5. Try with a smaller dataset first

### System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 5GB free disk space
- CPU-only execution

**Recommended:**
- Python 3.9+
- 16GB RAM
- 10GB free disk space
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.0+

### Best Practices

1. **Data Management**: Keep original data separate from processed data
2. **Experiment Tracking**: Use timestamps in result filenames
3. **Model Versioning**: Keep track of model versions and their performance
4. **Resource Monitoring**: Monitor GPU/CPU usage during training
5. **Result Backup**: Regularly backup your trained models and results
