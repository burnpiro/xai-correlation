import os
import torch
import random
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

from tqdm import tqdm
from data.datasets import (
    CustomDataset,
    get_default_transformation,
    get_inverse_normalization_transformation,
)
from models.resnet import (
    create_resnet18_model,
    create_resnet50_model,
)
from models.efficientnet import create_efficientnetb0_model
from models.densenet import create_densenet121_model
from models.common import NUM_OF_CLASSES, lime_mask
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import (
    IntegratedGradients,
    GuidedGradCam,
    Saliency,
    Deconvolution,
    GradientShap,
    GuidedBackprop,
    Lime,
)
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.metrics import (
    infidelity,
    sensitivity_max,
    infidelity_perturb_func_decorator,
)

default_cmap = LinearSegmentedColormap.from_list(
    "custom blue", [(0, "#ffffff"), (0.7, "#000000"), (1, "#000000")], N=256
)

METHODS = {
    "ig": "ig",
    "saliency": "saliency",
    "gradcam": "gradcam",
    "deconv": "deconv",
    "gradshap": "gradshap",
    "gbp": "gbp",
    # "lime": "lime",
}
torch.manual_seed(42)


def measure_rotation_model(
    model_version,
    dataset,
    out_folder,
    weights_dir,
    device,
    method=METHODS["gradcam"],
    sample_images=50,
    step=1,
    use_infidelity=False,
    use_sensitivity=False,
):
    invTrans = get_inverse_normalization_transformation()
    data_dir = os.path.join("data")

    if model_version == "resnet18":
        model = create_resnet18_model(num_of_classes=NUM_OF_CLASSES[dataset])
    elif model_version == "resnet50":
        model = create_resnet50_model(num_of_classes=NUM_OF_CLASSES[dataset])
    elif model_version == "densenet":
        model = create_densenet121_model(num_of_classes=NUM_OF_CLASSES[dataset])
    else:
        model = create_efficientnetb0_model(num_of_classes=NUM_OF_CLASSES[dataset])

    model.load_state_dict(torch.load(weights_dir))

    # print(model)

    model.eval()
    model.to(device)

    test_dataset = CustomDataset(
        dataset=dataset,
        transformer=get_default_transformation(),
        data_type="test",
        root_dir=data_dir,
        step=step,
        rotate=True,
    )
    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    try:
        image_ids = random.sample(
            range(0, test_dataset.__len__()), test_dataset.__len__()
        )
    except ValueError:
        raise ValueError(
            f"Image sample number ({test_dataset.__len__()}) exceeded dataset size ({test_dataset.__len__()})."
        )

    classes_map = test_dataset.classes_map

    print(f"Measuring {model_version} on {dataset} dataset, with {method}")
    print("-" * 10)
    pbar = tqdm(total=test_dataset.__len__(), desc="Model test completion")
    multipy_by_inputs = False
    if method == METHODS["ig"]:
        attr_method = IntegratedGradients(model)
        nt_samples = 1
        n_perturb_samples = 1
    if method == METHODS["saliency"]:
        attr_method = Saliency(model)
        nt_samples = 8
        n_perturb_samples = 2
    if method == METHODS["gradcam"]:
        if model_version == "efficientnet":
            attr_method = GuidedGradCam(model, model._conv_stem)
        elif model_version == "densenet":
            attr_method = GuidedGradCam(model, model.features.conv0)
        else:
            attr_method = GuidedGradCam(model, model.conv1)
        nt_samples = 8
        n_perturb_samples = 2
    if method == METHODS["deconv"]:
        attr_method = Deconvolution(model)
        nt_samples = 8
        n_perturb_samples = 2
    if method == METHODS["gradshap"]:
        attr_method = GradientShap(model)
        nt_samples = 8
        n_perturb_samples = 2
    if method == METHODS["gbp"]:
        attr_method = GuidedBackprop(model)
        nt_samples = 8
        n_perturb_samples = 2
    if method == "lime":
        attr_method = Lime(model)
        nt_samples = 8
        n_perturb_samples = 2
        feature_mask = torch.tensor(lime_mask).to(device)
        multipy_by_inputs = True
    if method == METHODS["ig"]:
        nt = attr_method
    else:
        nt = NoiseTunnel(attr_method)
    scores = []

    @infidelity_perturb_func_decorator(multipy_by_inputs=multipy_by_inputs)
    def perturb_fn(inputs):
        noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float()
        noise = noise.to(device)
        return inputs - noise

    for input, label in data_loader:
        pbar.update(1)
        inv_input = invTrans(input)
        input = input.to(device)
        input.requires_grad = True
        output = model(input)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        prediction_score = prediction_score.cpu().detach().numpy()[0][0]
        pred_label_idx.squeeze_()

        if use_sensitivity or use_infidelity:
            if method == METHODS["gradshap"]:
                baseline = torch.randn(input.shape)
                baseline = baseline.to(device)

        if method == "lime":
            attributions = attr_method.attribute(input, target=1, n_samples=50)
        elif method == METHODS["ig"]:
            attributions = nt.attribute(
                input,
                target=pred_label_idx,
                n_steps=25,
            )
        elif method == METHODS["gradshap"]:
            attributions = nt.attribute(
                input, target=pred_label_idx, baselines=baseline
            )
        else:
            attributions = nt.attribute(
                input,
                nt_type="smoothgrad",
                nt_samples=nt_samples,
                target=pred_label_idx,
            )

        if use_infidelity:
            infid = infidelity(
                model, perturb_fn, input, attributions, target=pred_label_idx
            )
            inf_value = infid.cpu().detach().numpy()[0]
        else:
            inf_value = 0

        if use_sensitivity:
            if method == "lime":
                sens = sensitivity_max(
                    attr_method.attribute,
                    input,
                    target=pred_label_idx,
                    n_perturb_samples=1,
                    n_samples=200,
                    feature_mask=feature_mask,
                )
            elif method == METHODS["ig"]:
                sens = sensitivity_max(
                    nt.attribute,
                    input,
                    target=pred_label_idx,
                    n_perturb_samples=n_perturb_samples,
                    n_steps=25,
                )
            elif method == METHODS["gradshap"]:
                sens = sensitivity_max(
                    nt.attribute,
                    input,
                    target=pred_label_idx,
                    n_perturb_samples=n_perturb_samples,
                    baselines=baseline,
                )
            else:
                sens = sensitivity_max(
                    nt.attribute,
                    input,
                    target=pred_label_idx,
                    n_perturb_samples=n_perturb_samples,
                )
            sens_value = sens.cpu().detach().numpy()[0]
        else:
            sens_value = 0

        if pbar.n in image_ids:
            rotation = test_dataset.data.iloc[pbar.n]["rotation"]
            attr_data = attributions.squeeze().cpu().detach().numpy()
            fig, ax = viz.visualize_image_attr_multiple(
                np.transpose(attr_data, (1, 2, 0)),
                np.transpose(inv_input.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                ["original_image", "heat_map"],
                ["all", "positive"],
                titles=["original_image", "heat_map"],
                cmap=default_cmap,
                show_colorbar=True,
                use_pyplot=False,
                fig_size=(8, 6),
            )
            if use_sensitivity or use_infidelity:
                ax[0].set_xlabel(
                    f"Infidelity: {'{0:.6f}'.format(inf_value)}\n Sensitivity: {'{0:.6f}'.format(sens_value)}"
                )
            fig.suptitle(
                f"True: {classes_map[str(label.numpy()[0])][0]}, Pred: {classes_map[str(pred_label_idx.item())][0]}\nScore: {'{0:.4f}'.format(prediction_score)}",
                fontsize=16,
            )
            fig.savefig(
                os.path.join(
                    out_folder,
                    f"{str(label.numpy()[0])}-rotation-{str(rotation)}-{classes_map[str(label.numpy()[0])][0]}-{classes_map[str(pred_label_idx.item())][0]}.png",
                )
            )
            plt.close(fig)
            # if pbar.n > 25:
            #     break

        scores.append([inf_value, sens_value])
    pbar.close()

    np.savetxt(
        os.path.join(out_folder, f"{model_version}-{dataset}-{method}.csv"),
        np.array(scores),
        delimiter=",",
        header="infidelity,sensitivity",
    )

    print(f"Artifacts stored at {out_folder}")
