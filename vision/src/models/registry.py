from typing import Callable
import torch
from ..params import ModelName, DEFAULT_DEVICE

MODEL_REGISTRY: dict[ModelName, Callable[[], torch.nn.Module]] = {}


def register_model(identifier: ModelName):
    def decorator(f):
        MODEL_REGISTRY[identifier] = f
        return f

    return decorator


def load_model(
    identifier: ModelName, device: torch.device = DEFAULT_DEVICE
) -> torch.nn.Module:
    return MODEL_REGISTRY[identifier]().to(device)
