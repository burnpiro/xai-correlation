# XAI

### Installation

Install Pytorch using [Local Installation Guide](https://pytorch.org/get-started/locally/). Then run:

### Dependencies (tested on Ubuntu 18.04):
- Python __3.8__
- torch __>=1.7.1__

```shell
pip install  -r requirements.txt
jupyter notebook
```

If you want to work on the same datasets download them using references in [Datasets Section](#datasets).

> ## For more detailed reproduction please follow the [WIKI](https://github.com/burnpiro/xai-correlation/wiki).

## Train and test all datasets for model

```shell
python train_and_test_all.py --model_version=resnet18
```

##### Parameters:
- `model_version`: version of the model [`resnet18`, `efficientnet`, `densenet`]

#### Saved Model output:
`models/saved_models/{model_version}-{dataset}-{train_skip}.pth`

For more training and testing options check out **[Train and Eval Models Wiki](https://github.com/burnpiro/xai-correlation/wiki/Train-and-Eval-Models)**

## Test augmentations

### Rotations

Generate attributions for rotated images. All experiments are stored in: `experiments/rotation/{dataset}/{model_version}-{train_skip}/{metohod}/...`

```shell
python test_rotation.py --model_version=resnet18 --dataset=edible-plants --train_skip=100%
```

### Filters

Generate attributions for images with applied filters. All experiments are stored in: `experiments/filters/{dataset}/{model_version}-{train_skip}/{metohod}/...`

```shell
python test_filters.py --model_version=resnet18 --dataset=edible-plants --train_skip=100%
```

#### Multiple settings

You can also paste multiple options:
```shell
python test_rotation.py --model_version=resnet18 --dataset=edible-plants --dataset=marvel --train_skip=100% --train_skip=80% --method=gradcam
```

This way you're going to measure results for `resnet18` base models, trained on `80%` and `100%` of `ediable-plants` and `marvel` datasets. Tests will be done using `Integrated Gradiens` and `GradCAM` methods. At the end you'll run `1 x 2 x 2 x 2 = 8 processes`.

##### Parameters:
- `model_version`: version of the model [`resnet18`, `efficientnet`, `densenet`]
- `dataset`: (optional) version of the dataset [`edible-plants`, `food101`, `marvel`, `plant-data`, `stanford-dogs`] if `None` then all versions are tested (`--weights` parameter is ignored)
- `train_skip`: (optional, default `None`) version of the train dataset size [`100%`, `80%`, `60%`, `40%`, `20%`], if `None` then all versions are tested (`--weights` parameter is ignored)
  - `method`: method to test [`ig`, `saliency`, `gradcam`, `deconv`, `gradshap`, `gbp`]
- `method`: method to test [`ig`, `saliency`, `gradcam`, `deconv`, `gradshap`, `gbp`]
- `weights`: (optional) path to `.pth` file with saved model, if none pasted then default one is used (`models/saved_models/{model_version}-{dataset}-{train_skip}.pth`)
- `weights`: (optional) path to `.pth` file with saved model, if none pasted then default one is used (`models/saved_models/{model_version}-{dataset}-{train_skip}.pth`)
- `use_infidelity`: (optional) if flag is set then Infidelity measure is calculated
- `use_sensitivity`: (optional) if flag is set then Sensitivity measure is calculated
- `render_results`: (optional) if flag is set then script renders images with attributions (otherwise only SSIM metric is calculated)

## Measure metrics for models

Calculate Infidelity and Sensitivity values for given model and dataset. Measures are calculated for every method available. All experiments are stored in: `experiments/{dataset}/{model_version}-{train_skip}/{metohod}/...`

```shell
python measure_model.py --model_version=resnet18 --dataset=edible-plants --train_skip=100%
```

You can also paste multiple options:
```shell
python measure_model.py --model_version=resnet18 --dataset=edible-plants --dataset=marvel --train_skip=100% --train_skip=80% --method=gradcam
```

This way you're going to measure results for `resnet18` base models, trained on `80%` and `100%` of `ediable-plants` and `marvel` datasets. Measurmenets will be done using `Integrated Gradiens` and `GradCAM` methods. At the end you'll run `1 x 2 x 2 x 2 = 8 processes`.

### WARNING!!!

If you want to calculate results for `Integrated Gradients`, make sure you have __enough memory on your GPU__. Test were run on __GTX 1080 Ti__ with 11GB memory available. That's not enough to calculate metrics for IG with the same number of perturbations as for other methods.

If you don't have enough memory run experiments without `ig` flag:
```shell
python measure_model.py --model_version=efficientnet --method=saliency --method=gradcam --method=deconv --method=gbp
```

##### Parameters:
- `model_version`: version of the model [`resnet18`, `efficientnet`, `densenet`]
- `dataset`: (optional) version of the dataset [`edible-plants`, `food101`, `marvel`, `plant-data`, `stanford-dogs`] if `None` then all versions are tested (`--weights` parameter is ignored)
- `train_skip`: (optional, default `None`) version of the train dataset size [`100%`, `80%`, `60%`, `40%`, `20%`], if `None` then all versions are tested (`--weights` parameter is ignored)
  - `method`: method to test [`ig`, `saliency`, `gradcam`, `deconv`, `gradshap`, `gbp`]
- `method`: method to test [`ig`, `saliency`, `gradcam`, `deconv`, `gradshap`, `gbp`]
- `weights`: (optional) path to `.pth` file with saved model, if none pasted then default one is used (`models/saved_models/{model_version}-{dataset}-{train_skip}.pth`)



## List of Notebooks

- `{Model} - Multiple metrics.ipynb` - All metrics for specific model.
- `{Model} - Score vs metrics.ipynb` - Comparison of metric vs score on dataset.
- `Metrics on default datasets.ipynb` - First version of metrics achieved on full datasets (not used in publication)
- `method_samples/Resnet18 IG NoiseTunnel.ipynb` - Integrated Gradients base explanation
- `method_samples/Resnet18 GradientShap.ipynb` - GradientShap base explanation
- `method_samples/Resnet18 Deconvolution.ipynb` - Deconvolution base explanation
- `method_samples/Resnet18 GBP.ipynb` - Guided Backpropagation base explanation
- `method_samples/Resnet18 Saliency.ipynb` - Saliency base explanation
- `dataset_eda/Stanford Dogs.ipynb` - EDA Stanford Dogs dataset
- `dataset_eda/Food 101.ipynb` - EDA Food 101 dataset
- `dataset_eda/Edible wild plants.ipynb` - EDA Edible wild plants dataset

## Example:

![IG Noise Tunnel](./img/ig_nt_result.png)

IG with Noise Tunnel

## References

* `IntegratedGradients`: [Axiomatic Attribution for Deep Networks, Mukund Sundararajan et al. 2017](https://arxiv.org/abs/1703.01365)
* `NoiseTunnel`: [Sanity Checks for Saliency Maps, Julius Adebayo et al. 2018](https://arxiv.org/abs/1810.03292)
* `Saliency`: [Deep Inside Convolutional Networks: Visualising
Image Classification Models and Saliency Maps, K. Simonyan, et. al. 2014](https://arxiv.org/pdf/1312.6034.pdf)
* `Deconvolution`: [Visualizing and Understanding Convolutional Networks, Scott Lundberg et al. 2017](https://arxiv.org/pdf/1705.07874.pdf)
* `GradientShap`: [A Unified Approach to Interpreting Model Predictions, Matthew D Zeiler et al. 2014](https://arxiv.org/pdf/1311.2901.pdf)
* `Guided Backpropagation`: [Striving for Simplicity: The All Convolutional Net, Jost Tobias Springenberg et al. 2015](https://arxiv.org/pdf/1412.6806.pdf)
* `Infidelity and Sensitivity`: [On the (In)fidelity and Sensitivity for Explanations](https://arxiv.org/abs/1901.09392)
* `ROAR`: [A Benchmark for Interpretability Methods in Deep Neural Networks](https://arxiv.org/abs/1806.10758)
* `SAM`: [SAM: The Sensitivity of Attribution Methods to Hyperparameters](https://arxiv.org/abs/2003.08754)

## Datasets

Extract all datasets into `./data` directory.

* `Stanford Dogs Dataset` - [Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset) (change folder name into `Stanford_Dogs_Dataset`)
![dogs](./img/dogs.png)
* `Food 101` - [Food 101 - Foodspotting](https://www.kaggle.com/kmader/food41)
![food101](./img/food101.png)
* `Edible wild plants` - [Edible wild plants](https://www.kaggle.com/gverzea/edible-wild-plants) (change folder name into `Edible_wild_plants`)
![wild](./img/wild-plants.png)
* `Plants_Dataset` - [Plants_Dataset[99 classes]](https://www.kaggle.com/muhammadjawad1998/plants-dataset99-classes?select=Plant_Data)
![wild](./img/plants.png)
* `Marvel Heroes` - [Marvel Heroes](https://www.kaggle.com/hchen13/marvel-heroes)
![wild](./img/marvel.png)