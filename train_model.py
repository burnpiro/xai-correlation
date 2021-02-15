import os
import torch

from absl import app, flags
from training.resnet_tune import train_resnet
from models.resnet import SPLIT_OPTIONS

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "model_version",
    "resnet18",
    ["resnet18", "resnet50"],
    "Model version [resnet18, resnet50]",
)
flags.DEFINE_enum(
    "dataset",
    "edible-plants",
    ["edible-plants", "food101", "marvel", "plant-data", "stanford-dogs"],
    "Dataset name [edible-plants, food101, marvel, plant-data, stanford-dogs]",
)

out_folder = os.path.join("models", "saved_models")
data_dir = os.path.join("data")


def main(_argv):
    for label, skip in SPLIT_OPTIONS.items():
        print(
            f"Training {FLAGS.model_version} model with {label} of data ({FLAGS.dataset})"
        )
        trained_model = train_resnet(
            FLAGS.dataset, FLAGS.model_version, data_dir=data_dir, skip=skip
        )

        print(
            f"Saving model to '{os.path.join(out_folder, f'{FLAGS.model_version}-{FLAGS.dataset}-{label}.pth')}'"
        )
        torch.save(
            trained_model.state_dict(),
            os.path.join(
                out_folder, f"{FLAGS.model_version}-{FLAGS.dataset}-{label}.pth"
            ),
        )


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
