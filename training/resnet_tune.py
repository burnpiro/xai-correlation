from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms

from training.helpers import train_model
from models.resnet import create_resnet18_model, create_resnet50_model
from data.datasets import CustomDataset, get_default_transformation

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_resnet(dataset, model_type="resnet18", data_dir=None):
    data_transform = get_default_transformation()
    train_dataset = CustomDataset(
        dataset=dataset,
        transformer=data_transform,
        data_type="train",
        root_dir=data_dir
    )
    test_dataset = CustomDataset(
        dataset=dataset,
        transformer=data_transform,
        data_type="test",
        root_dir=data_dir
    )

    # def show_image(image, label, dataset):
    #     print(f"Label: {label}")
    #     plt.imshow(image.permute(1,2,0))
    #     plt.show()
    #
    #
    # show_image(*train_dataset[0], train_dataset)

    dataset_loaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True, num_workers=4
        ),
        "val": torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4
        ),
    }

    print(f"Num of classes {len(train_dataset.classes)}")
    if model_type == "resnet18":
        model_ft = create_resnet18_model(num_of_classes=len(train_dataset.classes))
    else:
        model_ft = create_resnet50_model(num_of_classes=len(train_dataset.classes))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        model_ft,
        dataset_loaders,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        device,
        num_epochs=5,
    )

    return model_ft



# visualize_model(model_ft, dataset_loaders, device, class_names=train_dataset.classes)
