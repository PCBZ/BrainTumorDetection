from enum import Enum

class ModelType(Enum):
    VGG16 = "vgg16"
    RESNET50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"

class LearningStrategy(Enum):
    FEATURE_EXTRACTION = "feature_extraction"
    FINE_TUNING = "fine_tuning"
