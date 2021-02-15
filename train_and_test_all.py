import os
import torch

from absl import app, flags
from models.evaluation_helpers import test_model
from training.resnet_tune import train_resnet
from data.datasets import DATASETS
from models.resnet import SPLIT_OPTIONS

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "model_version",
    "resnet18",
    ["resnet18", "resnet50"],
    "Model version [resnet18, resnet50]",
)

model_folder = os.path.join("models", "saved_models")
out_folder = os.path.join("models", "evals")
data_dir = os.path.join("data")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(_argv):
    for dataset in DATASETS.keys():
        for label, skip in SPLIT_OPTIONS.items():
            print(
                f"Training {FLAGS.model_version} model with {label} of data ({dataset})"
            )
            trained_model = train_resnet(
                dataset, FLAGS.model_version, data_dir=data_dir, skip=skip
            )

            print(
                f"Saving model to '{os.path.join(model_folder, f'{FLAGS.model_version}-{dataset}-{label}.pth')}'"
            )
            torch.save(
                trained_model.state_dict(),
                os.path.join(
                    model_folder, f"{FLAGS.model_version}-{dataset}-{label}.pth"
                ),
            )

            print(
                f"Testing {FLAGS.model_version} model trained on {label} of data ({dataset})"
            )
            weights_dir = os.path.join(
                model_folder, f"{FLAGS.model_version}-{dataset}-{label}.pth"
            )
            if FLAGS.weights is not None:
                weights_dir = FLAGS.weights

            test_model(
                FLAGS.model_version, dataset, out_folder, weights_dir, device, label
            )


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
