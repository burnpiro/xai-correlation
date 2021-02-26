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
from models.common import NUM_OF_CLASSES
from models.resnet import create_resnet18_model, create_resnet50_model


def run_eval(preds, labels, names):
    conf_matrix = confusion_matrix(labels, preds)
    class_report = classification_report(
        labels, preds, target_names=names, output_dict=True
    )

    return conf_matrix, class_report


def save_cm(cm, labels, filename, figsize=(40, 40), subtitle=""):
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
    fig.suptitle("\n".join(["Confusion Matrix", subtitle]), fontsize=42)
    ax.xaxis.label.set_size(36)
    ax.yaxis.label.set_size(36)
    ax.tick_params(axis="both", which="major", labelsize=26)
    sns.set(font_scale=2.0)
    sns.heatmap(cm, cmap="viridis", fmt="", ax=ax)
    plt.savefig(filename, transparent=True)


def classification_report_latex(report, filename="report.txt"):
    df = pd.DataFrame(report).transpose()
    with open(filename, "w") as tf:
        tf.write(df.to_latex(float_format="%.3f"))


def test_model(model_version, dataset, out_folder, weights_dir, device, version="100%"):
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
    scores = []
    print("-" * 30)
    pbar = tqdm(total=test_dataset.__len__(), desc="Model test completion")
    for input, label in data_loader:
        input = input.to(device)
        pbar.update(1)
        output = model(input)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        pred_label_idx.squeeze_()
        label_val = int(label.detach().numpy()[0])
        pred_val = int(pred_label_idx.detach().item())
        labels.append(label_val)
        preds.append(pred_val)
        scores.append(
            [
                label_val,
                pred_val,
                label_val == pred_val,
                prediction_score.detach().cpu().numpy()[0][0],
            ]
        )
    pbar.close()

    conf_matrix, class_report = run_eval(preds, labels, test_dataset.classes)
    f1 = "{0:.4f}".format(class_report["weighted avg"]["f1-score"])
    acc = "{0:.4f}".format(class_report["weighted avg"]["precision"])
    subtitle = f"F1: {f1}, Prec: {acc}"
    save_cm(
        conf_matrix,
        test_dataset.classes,
        os.path.join(out_folder, f"{model_version}-{dataset}-{version}.png"),
        subtitle=subtitle,
    )

    classification_report_latex(
        class_report,
        filename=os.path.join(out_folder, f"{model_version}-{dataset}-{version}.txt"),
    )

    with open(
        os.path.join(out_folder, f"{model_version}-{dataset}-{version}.csv"), "w"
    ) as tf:
        tf.write(f"{f1},{acc}")

    scores_df = pd.DataFrame(np.array(scores), columns=['true', 'pred', 'currect', 'score'])
    scores_df.to_csv(os.path.join(out_folder, f"{model_version}-{dataset}-{version}-scores.csv"))

    print(
        f'Artifacts stored at {os.path.join(out_folder, f"{model_version}-{dataset}-{version}")}.*'
    )
