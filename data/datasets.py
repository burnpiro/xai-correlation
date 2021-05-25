from __future__ import print_function, division
import os
import gmic
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
    DATASETS["edible-plants"]: "Edible_wild_plants/datasets",
    DATASETS["food101"]: "food101",
    DATASETS["marvel"]: "marvel/marvel",
    DATASETS["plant-data"]: "Plant_Data/Plant_Data",
    DATASETS["stanford-dogs"]: "Stanford_Dogs_Dataset/images/Images",
}


"""
These are max values of the attribution for each model, dataset and XAI method.
Values are used to calculate SSIM metric (range set)
"""
MAX_ATT_VALUES = {
    "resnet18": {
        "ig": {
            "edible-plants": 0.4219718961032256,
            "food101": 1.3068271111898995,
            "marvel": 0.6114985791971517,
            "plant-data": 0.8875898804991542,
            "stanford-dogs": 1.565911949753198,
        },
        "saliency": {
            "edible-plants": 0.047666743,
            "food101": 0.045500886,
            "marvel": 0.067406125,
            "plant-data": 0.09445625,
            "stanford-dogs": 0.06071159,
        },
        "gradcam": {
            "edible-plants": 4.7747493e-05,
            "food101": 5.173298e-05,
            "marvel": 2.2187825e-05,
            "plant-data": 8.525976e-05,
            "stanford-dogs": 6.481426e-05,
        },
        "deconv": {
            "edible-plants": 0.7846228,
            "food101": 0.42328534,
            "marvel": 0.4730295,
            "plant-data": 0.6825042,
            "stanford-dogs": 1.2834907,
        },
        "gradshap": {
            "edible-plants": 0.14540403,
            "food101": 0.09303524,
            "marvel": 0.10294731,
            "plant-data": 0.28507224,
            "stanford-dogs": 0.14786015,
        },
        "gbp": {
            "edible-plants": 0.07977496,
            "food101": 0.06918508,
            "marvel": 0.050991096,
            "plant-data": 0.12655339,
            "stanford-dogs": 0.15254599,
        },
    },
    "efficientnet": {
        "ig": {
            "edible-plants": 0.732469091840906,
            "food101": 5.059399982141479,
            "marvel": 0.8100544955941875,
            "plant-data": 1.5519129716309399,
            "stanford-dogs": 2.329165646284462,
        },
        "saliency": {
            "edible-plants": 0.0950807,
            "food101": 0.03225366,
            "marvel": 0.043658517,
            "plant-data": 0.14042623,
            "stanford-dogs": 0.13611607,
        },
        "gradcam": {
            "edible-plants": 2.8511127e-05,
            "food101": 5.288464e-06,
            "marvel": 7.162203e-06,
            "plant-data": 4.2028296e-05,
            "stanford-dogs": 3.7838756e-05,
        },
        "deconv": {
            "edible-plants": 0.07810705,
            "food101": 0.025042508,
            "marvel": 0.03060332,
            "plant-data": 0.055335294,
            "stanford-dogs": 0.071512155,
        },
        "gradshap": {
            "edible-plants": 0.46427202,
            "food101": 0.05429075,
            "marvel": 0.15411238,
            "plant-data": 0.28000546,
            "stanford-dogs": 0.35282397,
        },
        "gbp": {
            "edible-plants": 0.06873036,
            "food101": 0.037462484,
            "marvel": 0.03062273,
            "plant-data": 0.06607512,
            "stanford-dogs": 0.10210772,
        },
    },
    "densenet": {
        "ig": {
            "edible-plants": 1.0996446153964408,
            "food101": 2.577734753749909,
            "marvel": 0.9976569403028434,
            "plant-data": 0.8164210705052278,
            "stanford-dogs": 2.090753485803276,
        },
        "saliency": {
            "edible-plants": 0.06872059,
            "food101": 0.048571102,
            "marvel": 0.052243587,
            "plant-data": 0.06894232,
            "stanford-dogs": 0.14973614,
        },
        "gradcam": {
            "edible-plants": 0.00027889787,
            "food101": 0.00018727055,
            "marvel": 0.0005107261,
            "plant-data": 0.00039552862,
            "stanford-dogs": 0.0021067497,
        },
        "deconv": {
            "edible-plants": 4677.1177,
            "food101": 80.11916,
            "marvel": 3037.732,
            "plant-data": 1715.9393,
            "stanford-dogs": 2044.188,
        },
        "gradshap": {
            "edible-plants": 0.14207457,
            "food101": 0.11977849,
            "marvel": 0.16221623,
            "plant-data": 0.14339326,
            "stanford-dogs": 0.34582284,
        },
        "gbp": {
            "edible-plants": 0.40056002,
            "food101": 0.21894407,
            "marvel": 0.51246065,
            "plant-data": 0.7207382,
            "stanford-dogs": 1.5651536,
        },
    },
}


def get_default_transformation():
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_inverse_normalization_transformation():
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )


def get_edible_plants_data(base_path="data/Edible_wild_plants/datasets"):
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


def get_dogs_data(base_path="data/Stanford_Dogs_Dataset/images/Images"):
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
        step=1,
        skip=None,
        rotate=False,
        add_filters=False,
    ):
        """
        Args:
            dataset (string): one od the datasets
            root_dir (string): Directory with all the images.
            transformer (callable, optional): Optional transform to be applied
                on a sample.
            data_type (string): "train" or "test"
            step (int, optional): if different than 1 then data is iterated with given step
            skip (List[int], optional): if set then skipping every nth element
            rotate (boolean, optional): create sample rotation images
        """
        if dataset == DATASETS["edible-plants"]:
            train_df, test_df = get_edible_plants_data(
                os.path.join(root_dir, PATHS[dataset])
            )

        if dataset == DATASETS["food101"]:
            train_df, test_df = get_food101_data(os.path.join(root_dir, PATHS[dataset]))

        if dataset == DATASETS["marvel"]:
            train_df, test_df = get_marvel_data(os.path.join(root_dir, PATHS[dataset]))

        if dataset == DATASETS["plant-data"]:
            train_df, test_df = get_plants_data(os.path.join(root_dir, PATHS[dataset]))

        if dataset == DATASETS["stanford-dogs"]:
            train_df, test_df = get_dogs_data(os.path.join(root_dir, PATHS[dataset]))

        self.data = train_df if data_type == "train" else test_df
        self._classes_df = self.data.drop_duplicates(subset=["class"])
        if step != 1:
            self.data = self.data.iloc[::step, :]
        if skip is not None:
            for skip_val in skip:
                self.data = self.data.drop(self.data.iloc[::skip_val].index, 0)
        self.transformer = transformer

        self.rotate = rotate
        if rotate:
            new_df = pd.DataFrame()
            for rotation in [0, -30, -15, 15, 30]:
                df1 = self.data.copy()
                df1["rotation"] = str(rotation)
                new_df = pd.concat([new_df, df1], ignore_index=False)
                new_df = new_df.reset_index().sort_values(by=['image', 'index'], ignore_index=True).set_index('index')
                new_df = new_df.reset_index(drop=True)

            self.data = new_df

        self.add_filters = add_filters
        if add_filters:
            new_df = pd.DataFrame()

            for filter in [
                "none",
                "fx_freaky_details 2,10,1,11,0,32,0",
                "normalize_local 8,10",
                "fx_boost_chroma 90,0,0",
                "fx_mighty_details 25,1,25,1,11,0",
                "sharpen 300",
            ]:
                df1 = self.data.copy()
                df1["filter"] = str(filter)
                new_df = pd.concat([new_df, df1], ignore_index=False)
                new_df = new_df.reset_index().sort_values(by=['image', 'index'], ignore_index=True).set_index('index')
                new_df = new_df.reset_index(drop=True)

            self.data = new_df

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

    @property
    def classes_map(self):
        """
        Get mapping for predicted classes

        :return: Dict[int: List[string]] Map for all classes
        """
        return {str(i): [className] for i, className in enumerate(self.classes)}

    def __getitem__(self, idx):
        """
        :param idx: int
        :return: Tuple(array(3,244,244), int)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        img = Image.open(row["image"]).convert("RGB")

        if self.add_filters:
            if row["filter"] != "none":
                images = []
                gmic.run(f'"{row["image"]}" {row["filter"]}', images)
                img = images[0].to_PIL()

        if self.rotate:
            img = img.rotate(int(row["rotation"]))

        if self.transformer:
            img = self.transformer(img)

        return img, int(row["class"])
