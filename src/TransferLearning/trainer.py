import copy
from tqdm import tqdm
import torch

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """
    Train the model for a specified number of epochs.
    Args:
        model: The neural network model to train
        dataloaders: Dictionary containing 'train' and 'test' DataLoader objects
        criterion: Loss function
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train the model
        device: Device to run the training on ('cuda' or 'cpu')
    Returns:
        model: The trained model
        history: Dictionary containing training loss and accuracy history
    """
    best_model_wts = copy.deepcopy(model.state_dict())
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

            # Forward pass: compute model output
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

        if epoch_acc.item() > best_train_acc:
            best_train_acc = epoch_acc.item()
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, history