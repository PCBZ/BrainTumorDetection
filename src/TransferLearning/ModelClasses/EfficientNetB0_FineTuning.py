import torch.nn as nn
from torchvision import models

class EfficientNetB0_FineTuning(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4, pretrained=True):
        super().__init__()
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        # 不冻结参数
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )
        self.model = model

    def forward(self, x):
        return self.model(x) 