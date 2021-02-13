import os
import torch

from absl import app, flags
from models.evaluation_helpers import test_model
from data.datasets import DATASETS

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

out_folder = os.path.join("models", "saved_models")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(_argv):
    weights_dir = os.path.join(out_folder, f"{FLAGS.model_version}-{FLAGS.dataset}.pth")
    if FLAGS.weights is not None:
        weights_dir = FLAGS.weights

    test_model(FLAGS.model_version, FLAGS.dataset, out_folder, weights_dir, device)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
