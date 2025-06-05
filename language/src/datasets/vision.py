from typing import Literal

import torch.utils.data as data
import torchvision.transforms as T  # type: ignore[import-untyped]
import torchvision.datasets as datasets  # type: ignore[import-untyped]

from ..params import DatasetName, DATA_DIR, DatasetSplit

from .registry import register_dataset


RESIZE, CROP = 256, 224
TRANSFORM_BASE = T.Compose(
    [
        T.Resize(RESIZE),
        T.CenterCrop(CROP),
        T.ToTensor(),
    ]
)
NORMALIZE_RGB = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

TRANSFORM_RGB = T.Compose([TRANSFORM_BASE, NORMALIZE_RGB])
TRANSFORM_GRAY = T.Compose(
    [TRANSFORM_BASE, T.Lambda(lambda x: x.repeat(3, 1, 1)), NORMALIZE_RGB]
)


@register_dataset(DatasetName.IMAGENET)
def get_imagenet(
    split: DatasetSplit,
    transform=TRANSFORM_RGB,
    target_transform=None,
    **kwargs,
):
    if len(kwargs) > 0:
        print(f"Unexpected kwargs: {kwargs.keys()}")

    dset = datasets.ImageNet(
        root=str(DATA_DIR / DatasetName.IMAGENET.value),
        split=split,
        transform=transform,
        target_transform=target_transform,
    )
    return dset


@register_dataset(DatasetName.CIFAR10)
def get_cifar10(
    split: DatasetSplit,
    transform=TRANSFORM_RGB,
    target_transform=None,
    num_classes: int | None = None,
    **kwargs,
):
    if len(kwargs) > 0:
        print(f"Unexpected kwargs: {kwargs.keys()}")

    return datasets.CIFAR10(
        root=str(DATA_DIR / DatasetName.CIFAR10.value),
        train=(split == "train"),
        transform=transform,
        target_transform=target_transform,
        download=True,
    )


@register_dataset(DatasetName.MNIST)
def get_mnist(
    split: DatasetSplit,
    transform=TRANSFORM_GRAY,
    target_transform=None,
    num_classes: int | None = None,
    **kwargs,
):
    if len(kwargs) > 0:
        print(f"Unexpected kwargs: {kwargs.keys()}")

    return datasets.MNIST(
        root=str(DATA_DIR / DatasetName.MNIST.value),
        train=(split == "train"),
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
