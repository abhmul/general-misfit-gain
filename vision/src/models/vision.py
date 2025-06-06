import torch
import torchvision  # type: ignore[import-untyped]

from ..params import *

from .registry import register_model


class ExtractEmbedding(torch.nn.Module):
    def __init__(self, head):
        super(ExtractEmbedding, self).__init__()
        self.head = head

    def forward(self, x):
        return x


def _alexnet_replace_fc(model):
    model.classifier = ExtractEmbedding(model.classifier)
    return model


@register_model(ModelName.RESNET50_DINO)
def resnet50_dino():
    model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    return model


@register_model(ModelName.VITB8_DINO)
def vitb8_dino():
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
    return model


@register_model(ModelName.ALEXNET)
def alexnet():
    model = torchvision.models.alexnet(pretrained=True)
    return _alexnet_replace_fc(model)


def get_alexnet_classifier():
    model = torchvision.models.alexnet(pretrained=True)
    return model.classifier
