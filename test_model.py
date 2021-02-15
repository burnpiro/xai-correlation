import os
import torch

from absl import app, flags
from models.evaluation_helpers import test_model
from data.datasets import DATASETS
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
    list(DATASETS.keys()),
    f"Dataset name, one of available datasets: {list(DATASETS.keys())}",
)
flags.DEFINE_string(
    "weights",
    None,
    "(optional) Dataset path to saved model",
)

model_folder = os.path.join("models", "saved_models")
out_folder = os.path.join("models", "evals")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(_argv):
    for label, skip in SPLIT_OPTIONS.items():
        print(
            f"Testing {FLAGS.model_version} model trained on {label} of data ({FLAGS.dataset})"
        )
        weights_dir = os.path.join(
            model_folder, f"{FLAGS.model_version}-{FLAGS.dataset}-{label}.pth"
        )
        if FLAGS.weights is not None:
            weights_dir = FLAGS.weights

        test_model(
            FLAGS.model_version, FLAGS.dataset, out_folder, weights_dir, device, label
        )


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
