import torch.nn as nn
from torchvision import models

class ResNet50_FeatureExtraction(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4, pretrained=True):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )
        self.model = model

    def forward(self, x):
        return self.model(x) 