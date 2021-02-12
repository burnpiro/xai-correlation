from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile, Image
from torchvision import transforms, utils
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = {
    "edible-plants": "edible-plants",
    "food101": "food101",
    "marvel": "marvel",
    "plant-data": "plant-data",
    "stanford-dogs": "stanford-dogs",
}

PATHS = {
    DATASETS["edible-plants"]: "Edible wild plants/datasets",
    DATASETS["food101"]: "food101",
    DATASETS["marvel"]: "marvel/marvel",
    DATASETS["plant-data"]: "Plant_Data/Plant_Data",
    DATASETS["stanford-dogs"]: "Stanford Dogs Dataset/images/Images",
}


def get_edible_plants_data(base_path="data/Edible wild plants/datasets"):
    train_base = os.path.join(base_path, "dataset")

    train_df = pd.DataFrame(columns=["image", "label", "class"])
    classes = []
    i = 0
    for dir_name in os.listdir(train_base):
        className = "-".join(dir_name.split(" ")).lower()
        classes.append(className)

        images = []
        for class_dir in os.scandir(os.path.join(train_base, dir_name)):
            images.append([class_dir.path, className, i])

        train_class_df = pd.DataFrame(
            np.array(images), columns=["image", "label", "class"]
        )

        train_df = train_df.append(train_class_df)
        i += 1

    train_df = train_df.reset_index(drop=True)

    test_base = os.path.join(base_path, "dataset-test")

    test_df = pd.DataFrame(columns=["image", "label", "class"])
    for dir_name in os.listdir(test_base):
        className = "-".join(dir_name.split(" ")).lower()

        images = []
        for class_dir in os.scandir(os.path.join(test_base, dir_name)):
            images.append([class_dir.path, className, classes.index(className)])

        test_class_df = pd.DataFrame(
            np.array(images), columns=["image", "label", "class"]
        )

        test_df = test_df.append(test_class_df)
        i += 1
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def get_food101_data(base_path="data/food101"):
    classes = pd.read_csv(
        os.path.join(base_path, "meta/meta/classes.txt"),
        header=None,
        index_col=0,
    )
    labels = pd.read_csv(os.path.join(base_path, "meta/meta/labels.txt"), header=None)
    classes["map"] = labels[0].values
    classes["class"] = labels.index
    classes_to_labels_map = classes["map"].to_dict()
    classes_to_class_map = classes["class"].to_dict()

    # Create mapping function path_name => class and class label
    def label_from_folder_map(class_to_label_map):
        return lambda o: class_to_label_map[(o.split(os.path.sep))[-2]]

    label_from_folder_food_func = label_from_folder_map(classes_to_labels_map)
    class_from_folder_food_func = label_from_folder_map(classes_to_class_map)

    train_df = pd.read_csv(
        os.path.join(base_path, "meta/meta/train.txt"), header=None
    ).apply(lambda x: x + ".jpg")
    test_df = pd.read_csv(
        os.path.join(base_path, "meta/meta/test.txt"), header=None
    ).apply(lambda x: x + ".jpg")
    train_df = train_df.rename(columns={0: "image"})
    test_df = test_df.rename(columns={0: "image"})

    train_df["class"] = train_df["image"].apply(
        lambda x: class_from_folder_food_func(x)
    )
    train_df["label"] = train_df["image"].apply(
        lambda x: label_from_folder_food_func(x)
    )
    train_df["image"] = train_df["image"].apply(
        lambda x: os.path.join(base_path, "images", x)
    )

    test_df["class"] = test_df["image"].apply(lambda x: class_from_folder_food_func(x))
    test_df["label"] = test_df["image"].apply(lambda x: label_from_folder_food_func(x))
    test_df["image"] = test_df["image"].apply(
        lambda x: os.path.join(base_path, "images", x)
    )

    return train_df, test_df


def get_plants_data(base_path="data/Plant_Data/Plant_Data"):
    train_base = os.path.join(base_path, "train")

    train_df = pd.DataFrame(columns=["image", "label", "class"])
    classes = []
    i = 0
    for dir_name in os.listdir(train_base):
        className = "-".join(dir_name.split(" ")).lower()
        classes.append(className)

        images = []
        for class_dir in os.scandir(os.path.join(train_base, dir_name)):
            images.append([class_dir.path, className, i])

        train_class_df = pd.DataFrame(
            np.array(images), columns=["image", "label", "class"]
        )

        train_df = train_df.append(train_class_df)
        i += 1

    train_df = train_df.reset_index(drop=True)

    test_base = os.path.join(base_path, "test")

    test_df = pd.DataFrame(columns=["image", "label", "class"])
    for dir_name in os.listdir(test_base):
        className = "-".join(dir_name.split(" ")).lower()

        images = []
        for class_dir in os.scandir(os.path.join(test_base, dir_name)):
            images.append([class_dir.path, className, classes.index(className)])

        test_class_df = pd.DataFrame(
            np.array(images), columns=["image", "label", "class"]
        )

        test_df = test_df.append(test_class_df)
        i += 1
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def get_marvel_data(base_path="data/marvel/marvel"):
    train_base = os.path.join(base_path, "train")

    train_df = pd.DataFrame(columns=["image", "label", "class"])
    classes = []
    i = 0
    for dir_name in os.listdir(train_base):
        className = "-".join(dir_name.split(" ")).lower()
        classes.append(className)

        images = []
        for class_dir in os.scandir(os.path.join(train_base, dir_name)):
            images.append([class_dir.path, className, i])

        train_class_df = pd.DataFrame(
            np.array(images), columns=["image", "label", "class"]
        )

        train_df = train_df.append(train_class_df)
        i += 1

    train_df = train_df.reset_index(drop=True)

    test_base = os.path.join(base_path, "valid")

    test_df = pd.DataFrame(columns=["image", "label", "class"])
    for dir_name in os.listdir(test_base):
        className = "-".join(dir_name.split(" ")).lower()

        images = []
        for class_dir in os.scandir(os.path.join(test_base, dir_name)):
            images.append([class_dir.path, className, classes.index(className)])

        test_class_df = pd.DataFrame(
            np.array(images), columns=["image", "label", "class"]
        )

        test_df = test_df.append(test_class_df)
        i += 1
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def get_dogs_data(base_path="data/Stanford Dogs Dataset/images/Images"):
    train_df = pd.DataFrame(columns=["image", "label", "class"])
    test_df = pd.DataFrame(columns=["image", "label", "class"])
    classes = []
    i = 0
    for dir_name in os.listdir(base_path):
        className = "-".join(dir_name.split("-")[1:])
        classes.append(className)

        images = []
        for class_dir in os.scandir(os.path.join(base_path, dir_name)):
            images.append([class_dir.path, className, i])

        num_of_images = len(images)
        train_class_df = pd.DataFrame(
            np.array(images[: int(num_of_images * 0.8)]),
            columns=["image", "label", "class"],
        )
        test_class_df = pd.DataFrame(
            np.array(images[int(num_of_images * 0.8) :]),
            columns=["image", "label", "class"],
        )

        train_df = train_df.append(train_class_df)
        test_df = test_df.append(test_class_df)
        i += 1

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


class CustomDataset(Dataset):
    """Edible wild plants dataset."""

    def __init__(
        self,
        dataset=DATASETS["edible-plants"],
        transformer=None,
        data_type="train",
        root_dir=None,
    ):
        """
        Args:
            dataset (string): one od the datasets
            root_dir (string): Directory with all the images.
            transformer (callable, optional): Optional transform to be applied
                on a sample.
            data_type (string): "train" or "test"
        """
        if dataset == DATASETS["edible-plants"]:
            train_df, test_df = get_edible_plants_data(
                os.path.join(root_dir, PATHS[dataset])
            )

        if dataset == DATASETS["food101"]:
            train_df, test_df = get_food101_data(
                os.path.join(root_dir, PATHS[dataset])
            )

        if dataset == DATASETS["marvel"]:
            train_df, test_df = get_marvel_data(
                os.path.join(root_dir, PATHS[dataset])
            )

        if dataset == DATASETS["plant-data"]:
            train_df, test_df = get_plants_data(
                os.path.join(root_dir, PATHS[dataset])
            )

        if dataset == DATASETS["stanford-dogs"]:
            train_df, test_df = get_dogs_data(
                os.path.join(root_dir, PATHS[dataset])
            )

        self.data = train_df if data_type == "train" else test_df
        self.transformer = transformer
        self._classes_df = self.data.drop_duplicates(subset=["class"])

    def __len__(self):
        """
        :return: int
        """
        return len(self.data)

    @property
    def classes(self):
        """
        Get list of classes

        :return List[string]
        """
        return self._classes_df.label.unique()

    def __getitem__(self, idx):
        """
        :param idx: int
        :return: Tuple(array(3,244,244), int)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        img = Image.open(row["image"]).convert("RGB")

        if self.transformer:
            img = self.transformer(img)

        return img, int(row["class"])
