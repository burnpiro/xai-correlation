import os
import json

import torch
import gmic
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def get_labels():
    labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)

    return labels_path, idx_to_labels


def get_input_image(path = "../../img/resnet/swan-3299528_1280.jpg"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    img = Image.open(path)
    transformed_img = transform(img)
    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)
    return input, transformed_img


def get_rotated_input_image(path = "../../img/resnet/swan-3299528_1280.jpg"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    img = Image.open(path)
    img = img.rotate(15)
    transformed_img = transform(img)
    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)
    return input, transformed_img


def get_input_image_with_filter(path = "../../img/resnet/swan-3299528_1280.jpg"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    img = Image.open(path)
    images = []
    gmic.run(f'"{path}" normalize_local 8,10', images)
    img = images[0].to_PIL()
    transformed_img = transform(img)
    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)
    return input, transformed_img

def predict(model, input, labels_map):
    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = labels_map[str(pred_label_idx.item())][1]
    return predicted_label, pred_label_idx, prediction_score
