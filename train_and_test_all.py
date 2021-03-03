import os
import torch

from absl import app, flags
from models.evaluation_helpers import test_model
from training.resnet_tune import train_resnet
from training.efficientnet_tune import train_efficientnet
from training.densenet_tune import train_densenet
from data.datasets import DATASETS
from models.common import SPLIT_OPTIONS, AVAILABLE_MODELS

FLAGS = flags.FLAGS

flags.DEFINE_multi_enum(
    "model_version",
    ["resnet18"],
    AVAILABLE_MODELS,
    f"Model version {AVAILABLE_MODELS}",
)

model_folder = os.path.join("models", "saved_models")
out_folder = os.path.join("models", "evals")
data_dir = os.path.join("data")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(_argv):
    for model_version in FLAGS.model_version:
        for dataset in DATASETS.keys():
            for label, skip in SPLIT_OPTIONS.items():
                print(
                    f"Training {model_version} model with {label} of data ({dataset})"
                )
                if model_version in AVAILABLE_MODELS[:1]:
                    trained_model = train_resnet(
                        dataset, model_version, data_dir=data_dir, skip=skip
                    )

                if model_version == AVAILABLE_MODELS[2]:
                    trained_model = train_efficientnet(
                        dataset, model_version, data_dir=data_dir, skip=skip
                    )

                if model_version == AVAILABLE_MODELS[3]:
                    trained_model = train_densenet(
                        dataset, model_version, data_dir=data_dir, skip=skip
                    )

                print(
                    f"Saving model to '{os.path.join(model_folder, f'{model_version}-{dataset}-{label}.pth')}'"
                )
                torch.save(
                    trained_model.state_dict(),
                    os.path.join(
                        model_folder, f"{model_version}-{dataset}-{label}.pth"
                    ),
                )

                print(
                    f"Testing {model_version} model trained on {label} of data ({dataset})"
                )
                weights_dir = os.path.join(
                    model_folder, f"{model_version}-{dataset}-{label}.pth"
                )

                test_model(
                    model_version, dataset, out_folder, weights_dir, device, label
                )


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
