from data.datasets import DATASETS

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