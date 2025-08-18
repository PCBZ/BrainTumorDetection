# Brain Tumor Detection System

A comprehensive deep learning project for automated brain tumor detection from MRI images. This CS6410 final project implements multiple state-of-the-art approaches including custom CNN architectures, transfer learning, and advanced class imbalance handling techniques.

![Brain Tumor Detection](https://img.shields.io/badge/PyTorch-Medical%20AI-red)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)

## 🧠 Project Overview

This project develops an intelligent system for detecting brain tumors from MRI scans using advanced deep learning techniques. The system addresses the critical challenge of accurate brain tumor detection in medical imaging through multiple complementary approaches:

### Key Features
- **Multi-Modal Approach**: Implements three distinct methodologies for comprehensive analysis
- **Medical Image Processing**: Specialized preprocessing and augmentation techniques for brain MRI data
- **Class Imbalance Handling**: Advanced techniques including Focal Loss and weighted sampling
- **Transfer Learning**: Leverages pre-trained models (VGG16, ResNet50, EfficientNet-B0) for enhanced performance
- **Robust Evaluation**: Cross-validation with stratified splitting and comprehensive metrics
- **Automated Pipeline**: End-to-end data processing from Kaggle dataset to model evaluation

### Technology Stack
- **Deep Learning Framework**: PyTorch
- **Pre-trained Models**: VGG16, ResNet50, EfficientNet-B0
- **Data Processing**: scikit-learn, NumPy, PIL
- **Visualization**: Matplotlib, Seaborn
- **Data Source**: Kaggle Brain MRI Dataset
- **Development Environment**: Python 3.8+, Jupyter Notebooks

### Approaches Implemented

1. **Custom CNN Architecture** (`src/CustomCNN/`)
   - Purpose-built convolutional neural network optimized for brain MRI analysis
   - Advanced regularization techniques including dropout and batch normalization
   - Configurable architecture with medical imaging-specific design choices

2. **Transfer Learning** (`src/TransferLearning/`)
   - Fine-tuning of state-of-the-art pre-trained models
   - Feature extraction and full model fine-tuning strategies
   - Comparative analysis across multiple architectures

3. **Class Imbalance Handling** (`src/ClassImbalanceHandling/`)
   - Data-centric deep learning approach
   - Advanced loss functions (Focal Loss, Class-Balanced Loss)
   - Medical imaging-specific data augmentation techniques

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- Internet connection for dataset download

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **GPU Memory**: 4GB+ VRAM (for training large models)
- **Storage**: 2GB+ free space for dataset and models

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/PCBZ/BrainTumorDetection.git
   cd BrainTumorDetection
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Using conda
   conda create -n brain_tumor_env python=3.9
   conda activate brain_tumor_env
   
   # Using venv
   python -m venv brain_tumor_env
   source brain_tumor_env/bin/activate  # Linux/macOS
   # or
   brain_tumor_env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   python -c "import torchvision; print('Torchvision version:', torchvision.__version__)"
   ```

### Environment Setup

1. **Configure Kaggle API** (for dataset download)
   ```bash
   pip install kaggle
   # Place your kaggle.json file in ~/.kaggle/ directory
   ```

2. **GPU Setup** (if available)
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

## 📖 Usage Instructions

### Quick Start

1. **Basic Model Training**
   ```python
   from src.data_utils import DataUtils
   from src.CustomCNN.CNN_brain_tumor import BrainTumorCNN
   from src.CustomCNN.CNN_evaluator import BrainTumorTrainer
   
   # Load data
   dataloaders, class_weights, datasets = DataUtils.load_brain_tumor_data_pipeline()
   
   # Initialize model and trainer
   model = BrainTumorCNN(num_classes=2, dropout_rate=0.3)
   trainer = BrainTumorTrainer(model)
   
   # Train model
   trained_model, metrics = trainer.run_complete_training_pipeline(dataloaders)
   ```

2. **Transfer Learning Approach**
   ```python
   from src.TransferLearning.transfer_learning_model import TransferLearningModel
   from src.TransferLearning.defines import ModelType, LearningStrategy
   
   # Initialize transfer learning model
   model = TransferLearningModel(
       model_type=ModelType.RESNET50,
       learning_strategy=LearningStrategy.FINE_TUNING
   )
   ```

3. **Class Imbalance Handling**
   ```python
   # Run the data-centric approach
   python src/ClassImbalanceHandling/approach3.py
   ```

### Input/Output Specifications

**Input:**
- **Format**: Brain MRI images (JPEG, PNG)
- **Dimensions**: Automatically resized to 224x224 pixels
- **Channels**: RGB (3 channels)
- **Dataset Structure**:
  ```
  data/
  ├── train/
  │   ├── yes/    # Images with tumors
  │   └── no/     # Images without tumors
  └── test/
      ├── yes/
      └── no/
  ```

**Output:**
- **Classification**: Binary (Tumor/No Tumor)
- **Confidence Score**: Probability between 0 and 1
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Example Usage Scenarios

1. **Research and Development**
   ```bash
   # Run comprehensive evaluation
   python src/evaluation.py
   ```

2. **Model Comparison**
   ```python
   # Compare all three approaches
   python -c "
   from src.evaluation import *
   # This will run all models and generate comparison plots
   "
   ```

3. **Custom Training**
   ```python
   # Train with custom parameters
   config = {
       'model_params': {
           'num_classes': 2,
           'dropout_rate': 0.4
       },
       'training_params': {
           'epochs': 50,
           'lr': 0.001,
           'batch_size': 32
       }
   }
   ```

## 🤖 Model Information

### Architecture Details

#### Custom CNN Model
- **Layers**: 6 convolutional layers with batch normalization
- **Activation**: ReLU activation functions
- **Regularization**: Dropout (0.3-0.4) and L2 regularization
- **Pooling**: MaxPooling and AdaptiveAvgPool
- **Final Layer**: Binary classification output

#### Transfer Learning Models
| Model | Parameters | Pre-training | Fine-tuning Strategy |
|-------|------------|--------------|---------------------|
| VGG16 | 138M | ImageNet | Feature extraction + Fine-tuning |
| ResNet50 | 25.6M | ImageNet | Full model fine-tuning |
| EfficientNet-B0 | 5.3M | ImageNet | Hybrid approach |

### Training Data Information

- **Dataset Source**: [Kaggle Brain MRI Images](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Total Images**: ~3,000 brain MRI scans
- **Classes**: 
  - **Yes** (Tumor present): ~1,500 images
  - **No** (No tumor): ~1,500 images
- **Split Ratio**: 80% training, 20% testing
- **Augmentation**: Medical imaging-specific transformations
- **Preprocessing**: Normalization, resizing, tensor conversion

### Performance Metrics

#### Expected Performance Ranges
| Approach | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|----------|----------|-----------|--------|----------|---------|
| Custom CNN | 85-90% | 0.85-0.92 | 0.83-0.89 | 0.84-0.90 | 0.90-0.95 |
| Transfer Learning | 90-95% | 0.90-0.96 | 0.88-0.94 | 0.89-0.95 | 0.92-0.97 |
| Class Imbalance Methods | 88-93% | 0.87-0.94 | 0.86-0.92 | 0.87-0.93 | 0.91-0.96 |

*Note: Actual performance may vary based on data splits and training configurations.*

### Model Validation
- **Cross-Validation**: 5-fold stratified cross-validation
- **Metrics**: Comprehensive evaluation including confusion matrices
- **Robustness**: Tested across multiple random seeds
- **Medical Relevance**: Focus on minimizing false negatives (high recall)

## 📁 Project Structure

```
BrainTumorDetection/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── CS6140.ipynb                      # Jupyter notebook for CS6140
├── .github/
│   └── workflows/
│       └── pylint.yml                # CI/CD pipeline
├── docs/
│   ├── api_documentation.md          # API documentation
│   └── user_guide.md                 # User guide
├── results/                          # Training results and outputs
├── src/                              # Source code
│   ├── __init__.py                   # Package initialization
│   ├── data_utils.py                 # Data loading and preprocessing utilities
│   ├── evaluation.py                 # Model evaluation and comparison
│   ├── evaluator.py                  # General evaluation utilities
│   ├── model_driver.py               # Model execution driver
│   ├── quantization.py               # Model quantization utilities
│   ├── CustomCNN/                    # Custom CNN implementation
│   │   ├── CNN_brain_tumor.py        # Main CNN model definitions
│   │   └── CNN_evaluator.py          # CNN training and evaluation
│   ├── TransferLearning/             # Transfer learning approaches
│   │   ├── transfer_learning_model.py # Transfer learning wrapper
│   │   ├── trainer.py                # Training utilities
│   │   ├── defines.py                # Model and strategy definitions
│   │   ├── Approach2.ipynb           # Transfer learning notebook
│   │   └── ModelClasses/             # Specific model implementations
│   └── ClassImbalanceHandling/       # Class imbalance solutions
│       └── approach3.py              # Data-centric deep learning approach
```

### Key Files and Their Purposes

| File | Purpose | Key Components |
|------|---------|----------------|
| `data_utils.py` | Data pipeline management | Dataset download, preprocessing, train/test split |
| `evaluation.py` | Model comparison | Cross-model evaluation, performance visualization |
| `CNN_brain_tumor.py` | Custom neural networks | BrainTumorCNN, SimpleCNN, model factory |
| `transfer_learning_model.py` | Transfer learning | Pre-trained model integration, fine-tuning |
| `approach3.py` | Class imbalance handling | Focal loss, data augmentation, weighted sampling |
| `trainer.py` | Training orchestration | Training loops, optimization, checkpointing |

## 📦 Dependencies

### Core Dependencies
```python
# Deep Learning Framework
torch>=1.9.0                   # PyTorch deep learning framework
torchvision>=0.10.0           # Computer vision utilities

# Data Science & ML
numpy>=1.21.0                 # Numerical computing
scikit-learn>=1.0.2           # Machine learning utilities

# Data Handling
kagglehub>=0.1.0              # Kaggle dataset integration
Pillow>=8.3.0                 # Image processing

# Visualization
matplotlib>=3.4.0             # Plotting and visualization
seaborn>=0.11.0               # Statistical visualization
tqdm>=4.62.0                  # Progress bars

# Development
jupyter>=1.0.0                # Jupyter notebook support
pandas>=1.3.0                 # Data manipulation
```

### Optional Dependencies
```python
# Advanced Features
opencv-python>=4.5.0          # Computer vision operations
albumentations>=1.1.0         # Advanced data augmentation
```

### Development Dependencies
```python
# Code Quality
pylint                        # Code linting
black                         # Code formatting
pytest                        # Unit testing
```

### Version Compatibility
- **Python**: 3.8, 3.9, 3.10, 3.11 (3.12 compatible)
- **PyTorch**: 1.9+ (tested up to 2.0)
- **CUDA**: 11.0+ (for GPU acceleration)
- **Operating Systems**: Linux (Ubuntu 18.04+), macOS (10.15+), Windows 10+

## 🤝 Contributing Guidelines

We welcome contributions to improve the Brain Tumor Detection system! Here's how you can contribute:

### Getting Started

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/BrainTumorDetection.git
   cd BrainTumorDetection
   ```

2. **Set Up Development Environment**
   ```bash
   # Create development environment
   conda create -n brain_tumor_dev python=3.9
   conda activate brain_tumor_dev
   pip install -r requirements.txt
   
   # Install development dependencies
   pip install pylint black pytest
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Guidelines

#### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings for all functions and classes
- Maintain consistent indentation (4 spaces)

#### Code Quality
```bash
# Run linting
pylint src/

# Format code
black src/

# Run tests
python -m pytest tests/
```

#### Documentation
- Update README.md for significant changes
- Add inline comments for complex algorithms
- Update API documentation in `docs/`
- Include usage examples

### Contribution Areas

#### High Priority
- [ ] Performance optimization for large datasets
- [ ] Additional pre-trained model integration
- [ ] Enhanced data augmentation techniques
- [ ] Real-time inference optimization

#### Medium Priority
- [ ] Web interface for model deployment
- [ ] Additional evaluation metrics
- [ ] Multi-class tumor classification
- [ ] Model interpretability features

#### Low Priority
- [ ] Docker containerization
- [ ] Cloud deployment scripts
- [ ] Additional dataset support
- [ ] Performance benchmarking suite

### Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Update documentation
   - Add tests for new features
   - Check code style compliance

2. **PR Guidelines**
   - Use descriptive commit messages
   - Include clear PR description
   - Reference related issues
   - Add screenshots for UI changes

3. **Review Process**
   - Code review by maintainers
   - Automated CI/CD checks
   - Performance impact assessment
   - Documentation review

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

### Attribution
When using this project, please provide appropriate credit:
```
Brain Tumor Detection System
Copyright (c) 2025 PCBZ
Licensed under MIT License
```

## 👥 Contact & Author Information

### Project Maintainer
- **GitHub**: [@PCBZ](https://github.com/PCBZ)
- **Repository**: [PCBZ/BrainTumorDetection](https://github.com/PCBZ/BrainTumorDetection)

### Academic Information
- **Course**: CS6410 - Advanced Topics in Computer Science
- **Project Type**: Final Project
- **Institution**: [University/Institution Name]
- **Academic Year**: 2024-2025

### Support & Communication
- **Issues**: Report bugs and feature requests via [GitHub Issues](https://github.com/PCBZ/BrainTumorDetection/issues)
- **Discussions**: Join project discussions in [GitHub Discussions](https://github.com/PCBZ/BrainTumorDetection/discussions)
- **Pull Requests**: Submit contributions via [GitHub Pull Requests](https://github.com/PCBZ/BrainTumorDetection/pulls)

## 🌟 Additional Information

### Screenshots & Demo

*[Screenshots would be added here showing the model performance, training curves, and example predictions]*

### Future Improvements & Roadmap

#### Short-term Goals (Next 3 months)
- [ ] **Performance Optimization**: Improve inference speed by 20%
- [ ] **Model Ensemble**: Implement voting classifier combining all approaches
- [ ] **Web Interface**: Develop Flask/FastAPI web application for model deployment
- [ ] **Documentation**: Add comprehensive API documentation with examples

#### Medium-term Goals (3-6 months)
- [ ] **Multi-class Classification**: Extend to detect specific tumor types
- [ ] **3D Analysis**: Implement 3D CNN for volumetric MRI analysis
- [ ] **Uncertainty Quantification**: Add confidence intervals and uncertainty measures
- [ ] **Mobile Deployment**: Optimize models for mobile/edge deployment

#### Long-term Vision (6+ months)
- [ ] **Clinical Integration**: Develop DICOM-compatible interface
- [ ] **Federated Learning**: Implement privacy-preserving distributed training
- [ ] **Explainable AI**: Add GRAD-CAM and other interpretability features
- [ ] **Real-time Processing**: Implement streaming inference pipeline

### Research Applications

This project serves as a foundation for:
- **Medical AI Research**: Advanced deep learning in healthcare
- **Computer Vision**: Image classification and medical imaging
- **Class Imbalance**: Techniques for imbalanced medical datasets
- **Transfer Learning**: Domain adaptation in medical imaging
- **Model Comparison**: Systematic evaluation of different approaches

### Acknowledgments

- **Dataset**: [Navoneel Chakrabarty](https://www.kaggle.com/navoneel) for the Brain MRI Dataset
- **Frameworks**: PyTorch team for the excellent deep learning framework
- **Pre-trained Models**: ImageNet contributors and model developers
- **Community**: Open-source contributors and medical AI researchers
- **Institution**: Course instructors and academic support

### Citation

If you use this project in your research, please cite:
```bibtex
@misc{brain_tumor_detection_2025,
  title={Brain Tumor Detection System: A Multi-Approach Deep Learning Framework},
  author={PCBZ},
  year={2025},
  publisher={GitHub},
  url={https://github.com/PCBZ/BrainTumorDetection}
}
```

---

**⚡ Quick Links**
- [🚀 Quick Start](#-installation--setup)
- [📖 Usage Guide](#-usage-instructions)
- [🤖 Model Details](#-model-information)
- [🤝 Contributing](#-contributing-guidelines)
- [📄 License](#-license)

**📊 Project Stats**
![GitHub last commit](https://img.shields.io/github/last-commit/PCBZ/BrainTumorDetection)
![GitHub issues](https://img.shields.io/github/issues/PCBZ/BrainTumorDetection)
![GitHub stars](https://img.shields.io/github/stars/PCBZ/BrainTumorDetection)
![GitHub forks](https://img.shields.io/github/forks/PCBZ/BrainTumorDetection)
