import torch.nn as nn
from torchvision import models

class EfficientNetB0_FeatureExtraction(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),
        )
        self.model = model

    def forward(self, x):
        return self.model(x) 