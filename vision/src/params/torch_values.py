from typing import Optional, Protocol, Any
from enum import Enum

import torch
import torch.utils.data as data


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"


CPU = torch.device(DeviceType.CPU.value)
CUDA = torch.device(DeviceType.CUDA.value)
DEFAULT_DEVICE = CUDA if torch.cuda.is_available() else CPU
SEED = 42
CPU_GENERATOR = torch.Generator(device=CPU)


def get_new_seed() -> int:
    return int(torch.randint(0, 2**32 - 1, (1,), generator=CPU_GENERATOR).item())


GPU_GENERATOR: Optional[torch.Generator] = None
if torch.cuda.is_available():
    GPU_GENERATOR = torch.Generator(device=CUDA)


def set_seed(seed: int) -> None:
    CPU_GENERATOR.manual_seed(seed)
    if GPU_GENERATOR is not None:
        GPU_GENERATOR.manual_seed(get_new_seed())


set_seed(SEED)


def get_generator(device=DEFAULT_DEVICE) -> torch.Generator:
    device_type = DeviceType(device.type)
    if device_type == DeviceType.CPU:
        return CPU_GENERATOR
    elif device_type == DeviceType.CUDA:
        assert GPU_GENERATOR is not None
        return GPU_GENERATOR
    else:
        raise ValueError(f"Device {device} of type {device.type} not recognized.")


class IndexableTensorContainer(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> torch.Tensor: ...


class Dataset(data.Dataset[torch.Tensor], IndexableTensorContainer):
    pass
