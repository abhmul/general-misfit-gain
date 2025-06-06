from typing import Protocol, Callable, Sized
from dataclasses import dataclass

import torch.utils.data as data
from ..params import DatasetName, DatasetSplit, Dataset


# Currently only doing vision datasets. Can add more as needed
class DatasetConstructor(Protocol):
    def __call__(self, split: DatasetSplit, **kwargs) -> Dataset: ...


DATASET_REGISTRY: dict[DatasetName, DatasetConstructor] = {}


def register_dataset(
    identifier: DatasetName,
) -> Callable[[DatasetConstructor], DatasetConstructor]:
    def decorator(f):
        DATASET_REGISTRY[identifier] = f
        return f

    return decorator


def load_dataset(identifier: DatasetName, split: DatasetSplit = "train", **kwargs):
    return DATASET_REGISTRY[identifier](split=split, **kwargs)
