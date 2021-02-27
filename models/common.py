from data.datasets import DATASETS
import numpy as np

AVAILABLE_MODELS = ["resnet18", "resnet50", "efficientnet"]

NUM_OF_CLASSES = {
    DATASETS["edible-plants"]: 62,
    DATASETS["food101"]: 101,
    DATASETS["marvel"]: 8,
    DATASETS["plant-data"]: 99,
    DATASETS["stanford-dogs"]: 120,
}

SPLIT_OPTIONS = {
    "100%": None,
    "80%": [5],
    "60%": [4, 5],
    "40%": [3, 4, 5],
    "20%": [2, 3, 4, 5],
}

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
