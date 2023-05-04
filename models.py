import torch
from network.atloc import AtLoc, AtLocPlus
from torchvision import transforms, models

def load_model(weights_path=None):
    """
    Loads MobileNetV2 pre-trained on ImageNet from PyTorch's cloud.
    Modifies last layers to fit our pose regression problem.
    """
    # model

    feature_extractor = models.resnet34(pretrained=True)
    atloc = AtLoc(feature_extractor, droprate=0.5, pretrained=True, lstm=True)
    model = atloc
    return model
