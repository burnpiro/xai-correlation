import os
import torch

from absl import app, flags
from pathlib import Path
from data.datasets import DATASETS
from models.measure_helpers import measure_model

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
    list(DATASETS.keys()),
    f"Dataset name, one of available datasets: {list(DATASETS.keys())}",
)
flags.DEFINE_string(
    "weights",
    None,
    "(optional) Dataset path to saved model",
)

model_folder = os.path.join("models", "saved_models")
out_folder = os.path.join("experiments")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(_argv):
    weights_dir = os.path.join(model_folder, f"{FLAGS.model_version}-{FLAGS.dataset}.pth")
    if FLAGS.weights is not None:
        weights_dir = FLAGS.weights

    Path(os.path.join(out_folder, FLAGS.dataset, FLAGS.model_version)).mkdir(
        parents=True, exist_ok=True
    )
    measure_model(
        FLAGS.model_version,
        FLAGS.dataset,
        os.path.join(out_folder, FLAGS.dataset, FLAGS.model_version),
        weights_dir,
        device,
    )


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
