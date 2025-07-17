"""

"""

import os
import json
import torch
from enum import Enum
from google.colab import drive
from typing import Optional, Dict, Any

class ModelType(Enum):
    VGG16 = "vgg16"
    RESNET50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"

class Strategy(Enum):
    FEATURE_EXTRACTION = "feature_extraction"
    FINE_TUNING = "fine_tuning"

class ModelDriveManager:
    """
    Class to manage model training and evaluation.
    """
    def __init__(self, base_drive_path: str = '/content/drive/MyDrive/BrainTumorDetection_TeamShared'):
        """
        Initialize the ModelDriveManager with the base path for Google Drive.
        args:
            base_drive_path (str): The base path in Google Drive where models and data will be
        """
        self.base_drive_path = base_drive_path
        self.is_mounted = False
        self._ensure_drive_mounted()
    
    def _ensure_drive_mounted(self):
        """
        Ensure that Google Drive is mounted.
        """
        if not self.is_mounted:
            drive.mount('/content/drive')
            self.is_mounted = True
            approach1_path = os.path.join(self.base_drive_path, "approach1")
            approach2_path = os.path.join(self.base_drive_path, "approach2")
            os.makedirs(approach1_path, exist_ok=True)
            os.makedirs(approach2_path, exist_ok=True)
    
    def upload_approach1_model(self, model: torch.nn.Module, model_name: str) -> str:
        """
        Upload the trained model to Google Drive under Approach 1.
        args:
            model (torch.nn.Module): The trained PyTorch model to be saved.
            model_name (str): The name of the model file.
        returns:
            str: The path where the model is saved in Google Drive.
        """
        self._ensure_drive_mounted()

        approach1_path = os.path.join(self.base_drive_path, "approach1")
        model_path = os.path.join(approach1_path, f"{model_name}.pth")
        try:
            torch.save(model.state_dict(), model_path)
        except Exception as e:
            print(f"Error saving model: {e}")
            return ""
    
    def upload_approach2_model(self, model: torch.nn.Module, model_type: ModelType, strategy: Strategy) -> str:
        """
        Upload the trained model to Google Drive under Approach 2.
        args:
            model (torch.nn.Module): The trained PyTorch model to be saved.
            model_type (ModelType): The type of the model (e.g., VGG16, ResNet50).
            strategy (Strategy): The learning strategy used (e.g., feature extraction, fine-tuning).
        returns:
            str: The path where the model is saved in Google Drive.
        """
        self._ensure_drive_mounted()

        approach2_path = os.path.join(self.base_drive_path, "approach2")
        model_path = os.path.join(approach2_path, f"{model_type.value}-{strategy.value}.pth")
        try:
            torch.save(model.state_dict(), model_path)
        except Exception as e:
            print(f"Error saving model: {e}")
            return ""
        
        return model_path

    def download_apporach1_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        self._ensure_drive_mounted()
        approach1_path = os.path.join(self.base_drive_path, "approach1")
        model_path = os.path.join(approach1_path, f"{model_name}.pth")
        if not os.path.exists(model_path):
            print(f"Model {model_name} does not exist in Approach 1.")
            return None
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        return model_state_dict
    
    def download_approach2_model(self, model_type: ModelType, strategy: Strategy) -> Optional[Dict[str, Any]]:
        """
        Download the trained model from Google Drive under Approach 2.
        args:
            model_type (ModelType): The type of the model (e.g., VGG16, ResNet50).
            strategy (Strategy): The learning strategy used (e.g., feature extraction, fine-tuning).
        returns:
            Optional[Dict[str, Any]]: The state dictionary of the model if it exists, otherwise None.
        """
        self._ensure_drive_mounted()
        
        approach2_path = os.path.join(self.base_drive_path, "approach2")
        model_path = os.path.join(approach2_path, f"{model_type.value}-{strategy.value}.pth")
        if not os.path.exists(model_path):
            print(f"Model {model_type.value} with strategy {strategy.value} does not exist in Approach 2.")
            return None
        
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        return model_state_dict
    

if __name__ == "__main__":
    drive.mount('/content/drive')

    shared_dir = '/content/drive/MyDrive/BrainTumorDetection_TeamShared'
    os.makedirs(shared_dir, exist_ok=True)

    approaches = ['approach1', 'approach2', 'approach3']
    for approach in approaches:
        approach_dir = os.path.join(shared_dir, approach)
        os.makedirs(approach_dir, exist_ok=True)