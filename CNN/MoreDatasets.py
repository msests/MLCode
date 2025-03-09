import kagglehub
import shutil
import os
import torch
import torchvision
import json
from collections import OrderedDict


def fetch_dataset():
    target = os.path.join(os.path.dirname(__file__), "./data/imagenet100")
    if os.path.exists(target):
        print("Dataset is already downloaded.")
        return target

    target_zip = os.path.join(os.path.dirname(
        __file__), "./data/imagenet100.zip")
    bash = f"curl -L -o {target_zip} https://www.kaggle.com/api/v1/datasets/download/ambityga/imagenet100"
    # 执行bash
    os.system(bash)

    os.mkdir(target)

    # 将compressed解压
    os.system(f"unzip -q {target_zip} -d {target}")
    return target


def ImageNet100(train: bool = True, transform=None):
    target = fetch_dataset()

    ImageNet100ClassToIdx = {}
    with open(os.path.join(target, "Labels.json"), "r") as f:
        labels = json.load(f, object_pairs_hook=OrderedDict)
        idx = 0
        for k, v in labels.items():
            ImageNet100ClassToIdx[k] = idx
            idx = idx + 1

    class_name_of_each_set = []

    def get_offset_transform(base):
        return lambda offset: ImageNet100ClassToIdx[class_name_of_each_set[base][offset]]

    if train:
        datasets = []
        for dir in os.listdir(target):
            if dir.startswith("train"):
                ds = torchvision.datasets.ImageFolder(
                    root=os.path.join(target, dir),
                    transform=transform,
                    target_transform=get_offset_transform(datasets.__len__())
                )
                datasets.append(ds)
                class_name_of_each_set.append(ds.classes)
        return torch.utils.data.ConcatDataset(datasets)
    else:
        datasets = []
        for dir in os.listdir(target):
            if dir.startswith("val"):
                ds = torchvision.datasets.ImageFolder(
                    root=os.path.join(target, dir),
                    transform=transform,
                    target_transform=get_offset_transform(0)
                )
                datasets.append(ds)
                class_name_of_each_set.append(ds.classes)
        return torch.utils.data.ConcatDataset(datasets)
