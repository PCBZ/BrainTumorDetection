import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

    # Save input data to file
    reports_dir = os.path.join("results", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    input_data = {
        "y_true": np.array(y_true).tolist(),
        "y_pred": np.array(y_pred).tolist(), 
        "y_probs": np.array(y_probs).tolist()
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_input_data_{timestamp}.json"
    filepath = os.path.join(reports_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(input_data, f, indent=2)
    
    print(f"\nInput data saved to: {filepath}")

