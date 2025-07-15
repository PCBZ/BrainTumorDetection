import torch.nn as nn
from torchvision import models
from defines import ModelType, LearningStrategy

class TransferLearningModel:
    """
    A class to handle transfer learning models for brain tumor detection.
    """
    def __init__(self, model_type: ModelType, learning_strategy: LearningStrategy):
        """
        Initialize the transfer learning model with the specified type and strategy.
        """
        self.model_type = model_type
        self.learning_strategy = learning_strategy
        self.model = self._initialize_model()
    
    def _set_parameter_requires_grad(self, model, feature_extracting):
        """
        Freeze model parameters if feature extraction is enabled.
        """
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    
    def _initialize_model(self):
        """
        Initialize the model based on the model type.
        """
        feature_extract = self.learning_strategy == LearningStrategy.FEATURE_EXTRACTION

        if self.model_type == ModelType.VGG16:
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self._set_parameter_requires_grad(model, feature_extract)

            # Get the input size of last layer
            num_features = model.classifier[-1].in_features

            # Replace last layer
            model.classifier[-1] = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 2),
            )
        elif self.model_type == ModelType.RESNET50:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self._set_parameter_requires_grad(model, feature_extract)

            # Get input features of last layer
            num_features = model.fc.in_features

            # Replace last layer
            model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 2),
            )
        elif self.model_type == ModelType.EFFICIENTNET_B0:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self._set_parameter_requires_grad(model, feature_extract)

            num_features = model.classifier[-1].in_features

            model.classifier[-1] = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 2),
            )

        return model