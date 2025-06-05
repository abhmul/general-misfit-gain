from typing import Callable

import torch
import torch.nn.functional as F

Stat = Callable[[torch.Tensor], torch.Tensor]


def binary_or_multi_stat(binary: Stat, multi: Stat) -> Stat:
    def stat(x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2

        if x.shape[1] == 1:
            return binary(x)
        else:
            return multi(x)

    return stat


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    return -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)


def _binary_entropy(logits: torch.Tensor) -> torch.Tensor:
    prob = F.sigmoid(logits)
    return -torch.sum(
        prob * F.logsigmoid(logits) + (1 - prob) * F.logsigmoid(-logits),
        dim=1,
    )


entropy = binary_or_multi_stat(_binary_entropy, _entropy)
