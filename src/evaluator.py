import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def evaluate_model(y_true, y_pred, y_probs, model_name):
    """
    Evaluate the model performance using various metrics.
    """
    all_labels = y_true
    all_preds = y_pred
    all_probs = y_probs

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs[:, 1])

    print("\nEvaluation metrics:")
    print(f"\nAccuracy: {accuracy :.2%}")
    print(f"Precision: {precision: .2%}")
    print(f"Recall: {recall: .2%}")
    print(f"F1 Score: {f1: .2%}")
    print(f"AUC Score: {auc: .4f}")

    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_dir = os.path.join("results", "figures", f"{model_name}_{timestamp}")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # Draw confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Tumor', 'Tumor'], 
                yticklabels=['No Tumor', 'Tumor'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_filename = f"confusion_matrix.png"
    cm_filepath = os.path.join(evaluation_dir, cm_filename)
    plt.savefig(cm_filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to: {cm_filepath}")
    
    # Draw ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_filename = f"roc_curve.png"
    roc_filepath = os.path.join(evaluation_dir, roc_filename)
    plt.savefig(roc_filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ROC curve saved to: {roc_filepath}")

    # Save input data to file in the same evaluation directory
    input_data = {
        "y_true": np.array(y_true).tolist(),
        "y_pred": np.array(y_pred).tolist(), 
        "y_probs": np.array(y_probs).tolist()
    }
    
    filename = f"input_data.json"
    filepath = os.path.join(evaluation_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(input_data, f, indent=2)
    
    print(f"\nInput data saved to: {filepath}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc
    }

