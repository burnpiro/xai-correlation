import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from data.datasets import CustomDataset, get_default_transformation
from models.resnet import create_resnet18_model, create_resnet50_model, NUM_OF_CLASSES


def run_eval(preds, labels, names):
    conf_matrix = confusion_matrix(labels, preds)
    class_report = classification_report(
        labels, preds, target_names=names, output_dict=True
    )

    return conf_matrix, class_report


def save_cm(cm, labels, filename, figsize=(40, 40)):
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d" % (p, c)
            elif c == 0:
                annot[i, j] = "0"
            else:
                annot[i, j] = "%.1f%%\n%d" % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("Confusion Matrix", fontsize=42)
    ax.xaxis.label.set_size(36)
    ax.yaxis.label.set_size(36)
    ax.tick_params(axis="both", which="major", labelsize=26)
    sns.set(font_scale=2.0)
    sns.heatmap(cm, cmap="viridis", fmt="", ax=ax)
    plt.savefig(filename, transparent=True)


def classification_report_latex(report, filename="report.txt"):
    df = pd.DataFrame(report).transpose()
    with open(filename, "w") as tf:
        tf.write(df.to_latex())


def test_model(model_version, dataset, out_folder, weights_dir, device):
    data_dir = os.path.join("data")

    if model_version == "resnet18":
        model = create_resnet18_model(num_of_classes=NUM_OF_CLASSES[dataset])
    else:
        model = create_resnet50_model(num_of_classes=NUM_OF_CLASSES[dataset])

    model.load_state_dict(torch.load(weights_dir))
    model.eval()
    model.to(device)

    test_dataset = CustomDataset(
        dataset=dataset,
        transformer=get_default_transformation(),
        data_type="test",
        root_dir=data_dir,
    )
    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    preds = []
    labels = []
    print(f'Testing {model_version} on {dataset} dataset')
    print('-' * 10)
    pbar = tqdm(total=test_dataset.__len__(), desc="Model test completion")
    for input, label in data_loader:
        input = input.to(device)
        pbar.update(1)
        output = model(input)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        pred_label_idx.squeeze_()
        labels.append(label.numpy()[0])
        preds.append(pred_label_idx.item())
    pbar.close()

    conf_matrix, class_report = run_eval(preds, labels, test_dataset.classes)
    save_cm(
        conf_matrix,
        test_dataset.classes,
        os.path.join(out_folder, f"{model_version}-{dataset}.png"),
    )

    classification_report_latex(
        class_report,
        filename=os.path.join(out_folder, f"{model_version}-{dataset}.txt"),
    )

    print(f'Artifacts stored at {os.path.join(out_folder, f"{model_version}-{dataset}")}.*')
