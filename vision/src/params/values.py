from typing import Literal
from enum import Enum

EP = 1e-9


# Datasets
class DatasetName(Enum):
    AMAZON_POLARITY = "amazon_polarity"
    ANTHROPIC_HH = "anthropic_hh"
    BOOLQ = "boolq"
    COSMOS_QA = "cosmos_qa"
    SCIQ = "sciq"
    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    MNIST = "mnist"

    def __str__(self):
        return self.value


DatasetSplit = Literal["train", "val"]


# Models
class ModelName(Enum):
    DISTIL_GPT2 = "distilgpt2"
    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    GPT2_XLARGE = "gpt2-xl"
    QWEN_1_8B = "Qwen/Qwen-1_8B"
    QWEN_7B = "Qwen/Qwen-7B"
    ALEXNET = "alexnet"
    RESNET50_DINO = "resnet50_dino"
    VITB8_DINO = "vitb8_dino"

    def __str__(self):
        return self.value
