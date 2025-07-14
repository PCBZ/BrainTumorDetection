"""
Model definitions for brain tumor detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights, ResNet50_Weights, EfficientNet_B0_Weights

from .config import MODEL_CONFIG


class BrainTumorClassifier(nn.Module):
    """Base class for brain tumor classifiers."""
    
    def __init__(self, model_name, num_classes=4, pretrained=True):
        super(BrainTumorClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._create_model(pretrained)
        
    def _create_model(self, pretrained):
        """Create the base model architecture."""
        raise NotImplementedError("Subclasses must implement _create_model")
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_feature_layers(self):
        """Freeze feature extraction layers for transfer learning."""
        raise NotImplementedError("Subclasses must implement freeze_feature_layers")
    
    def unfreeze_feature_layers(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True


class VGG16Classifier(BrainTumorClassifier):
    """VGG16-based brain tumor classifier."""
    
    def _create_model(self, pretrained):
        if pretrained:
            model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            model = models.vgg16(weights=None)
        
        # Modify the classifier
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
        
        return model
    
    def freeze_feature_layers(self):
        """Freeze feature extraction layers."""
        for param in self.model.features.parameters():
            param.requires_grad = False


class ResNet50Classifier(BrainTumorClassifier):
    """ResNet50-based brain tumor classifier."""
    
    def _create_model(self, pretrained):
        if pretrained:
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)
        
        # Modify the final layer
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        return model
    
    def freeze_feature_layers(self):
        """Freeze feature extraction layers."""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False


class EfficientNetB0Classifier(BrainTumorClassifier):
    """EfficientNet-B0-based brain tumor classifier."""
    
    def _create_model(self, pretrained):
        if pretrained:
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        
        # Modify the classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        
        return model
    
    def freeze_feature_layers(self):
        """Freeze feature extraction layers."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False


def create_model(model_name, num_classes=4, pretrained=True):
    """Factory function to create models."""
    
    if model_name == 'vgg16':
        return VGG16Classifier(model_name, num_classes, pretrained)
    elif model_name == 'resnet50':
        return ResNet50Classifier(model_name, num_classes, pretrained)
    elif model_name == 'efficientnet_b0':
        return EfficientNetB0Classifier(model_name, num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """Get information about a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_name': model.model_name if hasattr(model, 'model_name') else 'Unknown'
    }
