import torch.nn as nn
from torchvision import models
from data.datasets import DATASETS

NUM_OF_CLASSES = {
    DATASETS["edible-plants"]: 62,
    DATASETS["food101"]: 101,
    DATASETS["marvel"]: 8,
    DATASETS["plant-data"]: 99,
    DATASETS["stanford-dogs"]: 120,
}


def create_resnet18_model(num_of_classes):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_of_classes)

    return model


def create_resnet50_model(num_of_classes):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_of_classes)

    return model
