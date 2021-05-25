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
    MAX_ATT_VALUES,
)
from models.resnet import (
    create_resnet18_model,
    create_resnet50_model,
)
from models.efficientnet import create_efficientnetb0_model
from models.densenet import create_densenet121_model
from models.common import NUM_OF_CLASSES, lime_mask
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage

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


def measure_filter_model(
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
    render=False
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
        add_filters=True,
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

    OUR_FILTERS = [
        "none",
        "fx_freaky_details 2,10,1,11,0,32,0",
        "normalize_local 8,10",
        "fx_boost_chroma 90,0,0",
        "fx_mighty_details 25,1,25,1,11,0",
        "sharpen 300",
    ]
    idx = 0
    filter_count = 0
    filter_attrs = {filter_name: [] for filter_name in OUR_FILTERS}
    predicted_main_class = 0
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
        if OUR_FILTERS[filter_count] == 'none':
            predicted_main_class = pred_label_idx.item()

        if method == METHODS["gradshap"]:
            baseline = torch.randn(input.shape)
            baseline = baseline.to(device)

        if method == "lime":
            attributions = attr_method.attribute(input, target=1, n_samples=50)
        elif method == METHODS["ig"]:
            attributions = nt.attribute(
                input,
                target=predicted_main_class,
                n_steps=25,
            )
        elif method == METHODS["gradshap"]:
            attributions = nt.attribute(
                input, target=predicted_main_class, baselines=baseline
            )
        else:
            attributions = nt.attribute(
                input,
                nt_type="smoothgrad",
                nt_samples=nt_samples,
                target=predicted_main_class,
            )

        if use_infidelity:
            infid = infidelity(
                model, perturb_fn, input, attributions, target=predicted_main_class
            )
            inf_value = infid.cpu().detach().numpy()[0]
        else:
            inf_value = 0

        if use_sensitivity:
            if method == "lime":
                sens = sensitivity_max(
                    attr_method.attribute,
                    input,
                    target=predicted_main_class,
                    n_perturb_samples=1,
                    n_samples=200,
                    feature_mask=feature_mask,
                )
            elif method == METHODS["ig"]:
                sens = sensitivity_max(
                    nt.attribute,
                    input,
                    target=predicted_main_class,
                    n_perturb_samples=n_perturb_samples,
                    n_steps=25,
                )
            elif method == METHODS["gradshap"]:
                sens = sensitivity_max(
                    nt.attribute,
                    input,
                    target=predicted_main_class,
                    n_perturb_samples=n_perturb_samples,
                    baselines=baseline,
                )
            else:
                sens = sensitivity_max(
                    nt.attribute,
                    input,
                    target=predicted_main_class,
                    n_perturb_samples=n_perturb_samples,
                )
            sens_value = sens.cpu().detach().numpy()[0]
        else:
            sens_value = 0

        # filter_name = test_dataset.data.iloc[pbar.n]["filter"].split(" ")[0]
        attr_data = attributions.squeeze().cpu().detach().numpy()
        if render:
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
                    f"{str(idx)}-{str(filter_count)}-{str(label.numpy()[0])}-{str(OUR_FILTERS[filter_count])}-{classes_map[str(label.numpy()[0])][0]}-{classes_map[str(pred_label_idx.item())][0]}.png",
                )
            )
            plt.close(fig)
        # if pbar.n > 25:
        #     break
        score_for_true_label = output.cpu().detach().numpy()[0][predicted_main_class]

        filter_attrs[OUR_FILTERS[filter_count]] = [
            np.moveaxis(attr_data, 0, -1),
            "{0:.8f}".format(score_for_true_label),
        ]

        data_range_for_current_set = MAX_ATT_VALUES[model_version][method][dataset]
        filter_count += 1
        if filter_count >= len(OUR_FILTERS):
            ssims = []
            for rot in OUR_FILTERS:
                ssims.append(
                    "{0:.8f}".format(
                        ssim(
                            filter_attrs["none"][0],
                            filter_attrs[rot][0],
                            win_size=11,
                            data_range=data_range_for_current_set,
                            multichannel=True,
                        )
                    )
                )
                ssims.append(filter_attrs[rot][1])

            scores.append(ssims)
            filter_count = 0
            predicted_main_class = 0
            idx += 1

    pbar.close()

    indexes = []

    for filter_name in OUR_FILTERS:
        indexes.append(str(filter_name) + "-ssim")
        indexes.append(str(filter_name) + "-score")
    np.savetxt(
        os.path.join(out_folder, f"{model_version}-{dataset}-{method}-ssim-with-range.csv"),
        np.array(scores),
        delimiter=";",
        fmt="%s",
        header=";".join([str(rot) for rot in indexes]),
    )

    print(f"Artifacts stored at {out_folder}")
