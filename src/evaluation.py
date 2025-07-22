# =========================================
# 1. Environment Setup
# =========================================
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 2. Dataset Download and Preprocessing
# =========================================
from data_utils import DataUtils

# Download and organize data
# dataloaders: dict of DataLoader, class_weights: tensor, image_datasets: dict of ImageFolder
# train_dataset and test_dataset are ImageFolder objects
# You can use train_dataset and test_dataset directly for custom DataLoader
# If you want to use Approach 3's data-centric methods, use train_dataset
# If you want to use the original pipeline, use train_dataset and test_dataset as below

dataloaders, class_weights, image_datasets = DataUtils.load_brain_tumor_data_pipeline(target_path='./data/brain_tumor_split')
train_dataset = image_datasets['train']
test_dataset = image_datasets['test']

# =========================================
# 3. Import All Model Classes
# =========================================
from TransferLearning.ModelClasses.VGG16_FeatureExtraction import VGG16_FeatureExtraction
from TransferLearning.ModelClasses.VGG16_FineTuning import VGG16_FineTuning
from TransferLearning.ModelClasses.ResNet50_FeatureExtraction import ResNet50_FeatureExtraction
from TransferLearning.ModelClasses.ResNet50_FineTuning import ResNet50_FineTuning
from TransferLearning.ModelClasses.EfficientNetB0_FeatureExtraction import EfficientNetB0_FeatureExtraction
from TransferLearning.ModelClasses.EfficientNetB0_FineTuning import EfficientNetB0_FineTuning
from CustomCNN.CustomCNN_Approach1 import CustomCNN_Approach1

model_classes = [
    VGG16_FeatureExtraction,
    VGG16_FineTuning,
    ResNet50_FeatureExtraction,
    ResNet50_FineTuning,
    EfficientNetB0_FeatureExtraction,
    EfficientNetB0_FineTuning,
    CustomCNN_Approach1
]

model_names = [
    "VGG16_FeatureExtraction",
    "VGG16_FineTuning",
    "ResNet50_FeatureExtraction",
    "ResNet50_FineTuning",
    "EfficientNetB0_FeatureExtraction",
    "EfficientNetB0_FineTuning",
    "CustomCNN_Approach1"
]

# =========================================
# 4. Define Training and Evaluation Utility Functions
# =========================================
from approach3 import (
    Approach3Trainer, create_approach3_config, 
    test_model_with_approach3_methods
)
from evaluator import evaluate_model

def train_and_evaluate_original(model_class, train_dataset, test_dataset, config):
    """
    Train and evaluate the model using the original data pipeline
    """
    # 1. Data augmentation (original pipeline, usually only simple transforms)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset.transform = transform
    test_dataset.transform = transform

    # 2. DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 3. Model
    model = model_class(**config['model_params']).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 5. Training
    for epoch in range(config['epochs']):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 6. Testing
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    return all_labels, all_preds, all_probs

# =========================================
# 5. Training and Evaluation with Original Data Pipeline
# =========================================
original_results = {}
config = create_approach3_config()
for model_class, model_name in zip(model_classes, model_names):
    print(f"\n==== Training {model_name} on original data ====")
    y_true, y_pred, y_probs = train_and_evaluate_original(model_class, train_dataset, test_dataset, config)
    metrics = evaluate_model(y_true, y_pred, y_probs, model_name + "_original")
    original_results[model_name] = metrics

# =========================================
# 6. Training and Evaluation with Approach 3 Data-Centric Methods
# =========================================
approach3_results = {}
for model_class, model_name in zip(model_classes, model_names):
    print(f"\n==== Training {model_name} with Approach 3 data-centric methods ====")
    results = test_model_with_approach3_methods(model_class, train_dataset, config)
    # Use the predictions from the last fold (or customize as needed)
    y_true = results['targets']
    y_pred = np.array(results['predictions'])
    y_probs = np.array(results['probabilities'])
    metrics = evaluate_model(y_true, y_pred, y_probs, model_name + "_approach3")
    approach3_results[model_name] = metrics

# =========================================
# 7. Results Comparison and Visualization
# =========================================
import pandas as pd

summary = []
for model_name in model_names:
    orig = original_results[model_name]
    appr = approach3_results[model_name]
    summary.append({
        'Model': model_name,
        'Original_Accuracy': orig['accuracy'],
        'Approach3_Accuracy': appr['accuracy'],
        'Original_AUC': orig['auc'],
        'Approach3_AUC': appr['auc']
    })

df = pd.DataFrame(summary)
print(df)

df.plot(x='Model', y=['Original_Accuracy', 'Approach3_Accuracy'], kind='bar', figsize=(12,6), title='Accuracy Comparison')
plt.show()
df.plot(x='Model', y=['Original_AUC', 'Approach3_AUC'], kind='bar', figsize=(12,6), title='AUC Comparison')
plt.show()