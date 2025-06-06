from typing import Literal
import torch
import torch.nn.functional as F
import torch.nn as nn

from .params import EP, CUDA

Precision = Literal["half", "single", "double"]
PRECISION_DTYPE_DICT: dict[Precision, torch.dtype] = {
    "half": torch.float16,
    "single": torch.float32,
    "double": torch.float64,
}

FLOATING_TYPES = [torch.float16, torch.float32, torch.float64]
SIGNED_TYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
UNSIGNED_TYPES = [torch.uint8, torch.bool]

SIGNED_INT_IINFOS = {dtype: torch.iinfo(dtype) for dtype in SIGNED_TYPES}
BINARY_DTYPE = torch.int8
PM_DTYPE = torch.int8


def get_index_type(max_val: int):
    if max_val <= SIGNED_INT_IINFOS[torch.int32].max:
        return torch.int32
    else:
        return torch.int64


def sigmoid_or_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.size(dim) == 1:
        return torch.sigmoid(x)
    else:
        return torch.softmax(x, dim=dim)


def logit_or_log(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.size(dim) == 1:
        return torch.logit(x)
    else:
        return torch.log(x)


def clamped_log(x: torch.Tensor, min_val=EP) -> torch.Tensor:
    return x.clamp(min=min_val).log_()


def clamped_logit(x: torch.Tensor, min_val=EP) -> torch.Tensor:
    return x.clamp(min=min_val, max=1 - min_val).logit_()


def rand_unit_vectors(
    n: int,
    d: int,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    x = torch.randn((n, d), device=device, generator=generator)
    x /= x.norm(dim=1, keepdim=True)
    return x


def hardmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.size(dim) == 1:
        return torch.ones_like(x)
    else:
        idx = torch.argmax(x, dim=dim, keepdim=True)
        ret = torch.zeros_like(x)
        ret.scatter_(dim, idx, 1)
        return ret


def expand_along_dim(x: torch.Tensor, dim: int, num_dims: int) -> torch.Tensor:
    return x.view((1,) * dim + (-1,) + (1,) * (num_dims - dim - 1))


def select_along_dim(x: torch.Tensor, dim: int, idx: torch.Tensor) -> torch.Tensor:
    assert x.dim() == idx.dim()

    select = []
    for i in range(x.dim()):
        if i == dim:
            select.append(idx)
        else:
            selector = torch.arange(x.size(i), device=x.device, dtype=idx.dtype)
            selector = expand_along_dim(selector, i, x.dim())
            select.append(selector)

    return x[tuple(select)]


def combinations(x: torch.Tensor, r: int = 2, dim: int = -1):
    # torch.combinations only works on 1D tesnor. Open issue to
    # add a batched version: https://github.com/pytorch/pytorch/issues/40375
    # For now use the following
    inds = torch.combinations(
        torch.arange(x.size(dim), dtype=get_index_type(x.size(dim)), device=x.device),
        r=r,
    )
    selector = (slice(None),) * dim + (inds,) + (slice(None),) * (x.dim() - dim - 1)
    return x[selector]


def base_2_accumulator(
    length: int,
    little_endian: bool = False,
    device: torch.device | None = None,
    dtype=None,
) -> torch.Tensor:
    """
    It is the responsibility of the caller to not pass a dtype
    that will overflow!

    Returns
    -------
    (length) torch.LongTensor
        An array of powers of 2. The order is decreasing if little_endian = False
    """
    if dtype is None:
        dtype = get_index_type(2 ** (length - 1))

    powers_of_2 = torch.bitwise_left_shift(
        1, torch.arange(length, dtype=dtype, device=device)
    )
    if little_endian:
        return powers_of_2
    else:
        return torch.flip(powers_of_2, dims=(0,))


def bitarray2dec(
    arr: torch.Tensor,
    little_endian=False,
    dim: int = -1,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = arr.device

    if dtype is None:
        dtype = get_index_type(2 ** arr.shape[dim])

    # CUDA only supports tensordot for double,float, half.
    if CUDA == device:
        intermediate_dtype = torch.float
    else:
        intermediate_dtype = dtype

    if arr.shape[dim] == 0:
        reduced_shape = arr.shape[:dim] + arr.shape[dim + 1 :]
        return torch.zeros(reduced_shape, dtype=dtype, device=device)

    base_2 = base_2_accumulator(
        arr.shape[dim],
        little_endian=little_endian,
        device=device,
        dtype=intermediate_dtype,
    )
    return torch.tensordot(arr.to(intermediate_dtype), base_2, dims=[[dim], [0]]).to(
        dtype
    )


class ModuleEnhanced(nn.Module):
    def __init__(self):
        super(ModuleEnhanced, self).__init__()

    def project_parameters(self) -> None:
        pass

    @staticmethod
    def enhance(x: nn.Module) -> "ModuleEnhanced":
        if isinstance(x, ModuleEnhanced):
            return x
        else:
            return _ModuleEnhancedWrapper(x)


class _ModuleEnhancedWrapper(ModuleEnhanced):
    def __init__(self, module: nn.Module):
        super(_ModuleEnhancedWrapper, self).__init__()
        assert not isinstance(module, ModuleEnhanced)
        self.module = module

    def forward(self, x):
        return self.module(x)


def project_to_simplex(x: torch.Tensor, dim: int) -> torch.Tensor:
    sorted_x, _ = torch.sort(x, dim=dim, descending=True)
    cumsum_sorted_x = torch.cumsum(sorted_x, dim=dim)
    range_tensor = expand_along_dim(
        torch.arange(1, x.size(dim) + 1, device=x.device, dtype=x.dtype),
        dim=dim,
        num_dims=x.dim(),
    )
    t = (cumsum_sorted_x - 1) / range_tensor
    k = torch.sum(sorted_x > t, dim=dim, keepdim=True)
    tau = torch.gather(t, dim, k - 1)
    return torch.clamp(x - tau, min=0)
