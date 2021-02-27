import os
import torch

from absl import app, flags
from models.evaluation_helpers import test_model
from data.datasets import DATASETS
from models.common import SPLIT_OPTIONS, AVAILABLE_MODELS

FLAGS = flags.FLAGS

flags.DEFINE_multi_enum(
    "model_version",
    ["resnet18"],
    AVAILABLE_MODELS,
    f"Model version {AVAILABLE_MODELS}",
)
flags.DEFINE_multi_enum(
    "dataset",
    ["edible-plants"],
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
    for model_version in FLAGS.model_version:
        for dataset in FLAGS.dataset:
            for label, skip in SPLIT_OPTIONS.items():
                print(
                    f"Testing {model_version} model trained on {label} of data ({dataset})"
                )
                weights_dir = os.path.join(
                    model_folder, f"{model_version}-{dataset}-{label}.pth"
                )
                if (
                    FLAGS.weights is not None
                    and len(FLAGS.model_version) == 0
                    and len(FLAGS.dataset) == 0
                ):
                    weights_dir = FLAGS.weights

                test_model(
                    model_version, dataset, out_folder, weights_dir, device, label
                )


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
