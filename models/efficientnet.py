from efficientnet_pytorch import EfficientNet
import torch.nn as nn

B0_MODEL_NAME = "efficientnet-b0"


def create_efficientnetb0_model(num_of_classes):
    model = EfficientNet.from_pretrained(B0_MODEL_NAME)
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, num_of_classes)

    return model
