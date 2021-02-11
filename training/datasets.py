from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils


def get_edible_plants_data(base_path):
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


class EdibleWildPlantsDataset(Dataset):
    """Edible wild plants dataset."""

    def __init__(
        self,
        root_dir="../data/Edible wild plants/datasets",
        transform=None,
        data_type="train",
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            data_type (string): "train" or "test"
        """
        train_df, test_df = get_edible_plants_data(root_dir)
        self.data = train_df if data_type == "train" else test_df
        self.root_dir = root_dir
        self.transform = transform
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
        img = Image.open(row["image"]).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, int(row["class"])
