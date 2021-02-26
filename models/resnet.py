import torch.nn as nn
import numpy as np
from torchvision import models

lime_mask = []
input_size = 224  # size of the model's input
scale_factor = 4  # input_size has to be dividable by scale factor

for row in range(0, int(input_size / scale_factor)):
    row_input = []
    for group in range(0, int(input_size / scale_factor)):
        for k in range(0, scale_factor):
            row_input.append(row * int(input_size / scale_factor) + group)

    for k in range(0, scale_factor):
        lime_mask.append(row_input)

lime_mask = np.array([lime_mask, lime_mask, lime_mask])


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
