import os
import torch

from absl import app, flags
from training.resnet_tune import train_resnet
from data.datasets import DATASETS
from models.resnet import SPLIT_OPTIONS

FLAGS = flags.FLAGS

flags.DEFINE_multi_enum(
    "model_version",
    ["resnet18"],
    ["resnet18", "resnet50"],
    "Model version [resnet18, resnet50]",
)
flags.DEFINE_multi_enum(
    "dataset",
    ["edible-plants"],
    list(DATASETS.keys()),
    f"Dataset name, one of available datasets: {list(DATASETS.keys())}",
)

out_folder = os.path.join("models", "saved_models")
data_dir = os.path.join("data")


def main(_argv):
    for model_version in FLAGS.model_version:
        for dataset in FLAGS.dataset:
            for label, skip in SPLIT_OPTIONS.items():
                print(
                    f"Training {model_version} model with {label} of data ({dataset})"
                )
                trained_model = train_resnet(
                    dataset, model_version, data_dir=data_dir, skip=skip
                )

                print(
                    f"Saving model to '{os.path.join(out_folder, f'{model_version}-{dataset}-{label}.pth')}'"
                )
                torch.save(
                    trained_model.state_dict(),
                    os.path.join(out_folder, f"{model_version}-{dataset}-{label}.pth"),
                )


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
