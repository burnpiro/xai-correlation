import torch.nn as nn
from torchvision import models

B0_MODEL_NAME = "densenet121"


def create_densenet121_model(num_of_classes):
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_of_classes)

    return model
