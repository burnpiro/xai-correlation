import os
import torch

from absl import app, flags
from pathlib import Path
from data.datasets import DATASETS
from models.common import SPLIT_OPTIONS, AVAILABLE_MODELS
from models.measure_helpers import measure_model, METHODS
import warnings

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS

flags.DEFINE_multi_enum(
    "model_version",
    ["resnet18"],
    AVAILABLE_MODELS,
    f"Model version {AVAILABLE_MODELS}",
)
flags.DEFINE_multi_enum(
    "dataset",
    [],
    list(DATASETS.keys()),
    f"(optional) Dataset name, one of available datasets: {list(DATASETS.keys())}",
)
flags.DEFINE_multi_enum(
    "train_skip",
    [],
    list(SPLIT_OPTIONS.keys()),
    f"(optional) version of the train dataset size: {list(SPLIT_OPTIONS.keys())}",
)
flags.DEFINE_multi_enum(
    "method",
    [],
    list(METHODS.keys()),
    f"(optional) select on of available methods: {list(METHODS.keys())}",
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
    datasets = DATASETS.keys()
    if len(FLAGS.dataset) > 0:
        datasets = FLAGS.dataset

    items = SPLIT_OPTIONS
    if len(FLAGS.train_skip) > 0:
        items = {
            train_skip: SPLIT_OPTIONS[train_skip] for train_skip in FLAGS.train_skip
        }

    methods = METHODS.values()
    if len(FLAGS.method) > 0:
        methods = [METHODS[method] for method in FLAGS.method]

    for model_version in FLAGS.model_version:
        for dataset in datasets:
            for label, skip in items.items():
                weights_dir = os.path.join(
                    model_folder,
                    f"{model_version}-{dataset}-{label}.pth",
                )
                if (
                    FLAGS.weights is not None
                    and len(FLAGS.train_skip) == 0
                    and len(FLAGS.dataset) == 0
                ):
                    weights_dir = FLAGS.weights

                for method in methods:
                    Path(
                        os.path.join(
                            out_folder,
                            dataset,
                            f"{model_version}-{label}",
                            method,
                        )
                    ).mkdir(parents=True, exist_ok=True)
                    step = 1
                    if dataset == DATASETS["food101"]:
                        step = 5
                    if dataset == DATASETS["stanford-dogs"]:
                        step = 2
                    if dataset == DATASETS["plant-data"]:
                        step = 2
                    measure_model(
                        model_version,
                        dataset,
                        os.path.join(
                            out_folder,
                            dataset,
                            f"{model_version}-{label}",
                            method,
                        ),
                        weights_dir,
                        device,
                        method=method,
                        step=step,
                    )


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
