import copy
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, roc_auc_score, recall_score, f1_score
import torch
import torch.nn as nn

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """
    Train model.
    """
    since = time.time()

    beset_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0

    history = {
        'train_loss': [],
        'train_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloaders['train'], desc='train'):
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass: compute model ouput
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass: update params
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.cpu().numpy())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_train_acc:
            best_train_acc = epoch_acc
            beset_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    model.load_state_dict(beset_model_wts)
    return model, history

def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate model.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass: get model predictions
            outputs = model(inputs)

            probs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())


    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Tumor', 'Tumor']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    accuracy_score = accuracy_score(all_labels, all_preds)
    precision_score = precision_score(all_labels, all_preds, zero_division=0)
    recall_score = recall_score(all_labels, all_preds, zero_division=0)
    f1_score = f1_score(all_labels, all_preds, zero_division=0)

    auc_score = roc_auc_score(all_labels, all_probs[:, 1])

    print("\nEvaluation metrics:")
    print(f"\nAccuracy: {accuracy_score :.2%}")
    print(f"Precision: {precision_score: .2%}")
    print(f"Recall: {recall_score: .4f}")
    print(f"F1 Score: {f1_score: .4f}")
    print(f"AUC Score: {auc_score: .4f}")

    return {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
        'auc_score': auc_score,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }