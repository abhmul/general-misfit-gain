from typing import Optional

import torch
import torch.nn as nn

from ..torch_utils import sigmoid_or_softmax, logit_or_log, ModuleEnhanced


class PrimalConvexCombination(ModuleEnhanced):
    def __init__(self, num_heads: int):
        super(PrimalConvexCombination, self).__init__()
        assert num_heads > 1

        self.num_heads = num_heads
        self.combine = nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, x):
        # Expects primal inputs of shape (batch_size, output_dim, num_heads)
        assert x.dim() == 3
        assert x.shape[2] == self.num_heads

        coefficients = torch.softmax(self.combine, dim=0)
        x = torch.einsum("bij,j->bi", x, coefficients)
        return x


class LogisticRegression(ModuleEnhanced):
    def __init__(
        self, input_dim: int, output_dim: int, num_heads: int = 1, logit_outputs=True
    ):
        super(LogisticRegression, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.logit_outputs = logit_outputs
        self.finetune = nn.Linear(input_dim, output_dim * self.num_heads)
        if self.num_heads > 1:
            self.combine: PrimalConvexCombination | None = PrimalConvexCombination(
                num_heads=self.num_heads
            )
        else:
            self.combine = None

    def forward(self, x):
        x = self.finetune(x)
        if self.num_heads > 1:
            assert self.combine is not None
            x = x.view(-1, self.output_dim, self.num_heads)
            # Convert to primal for convex combination
            x = sigmoid_or_softmax(x, dim=1)
            x = self.combine(x)
            if self.logit_outputs:
                x = logit_or_log(x, dim=1)
            return x
        else:
            if not self.logit_outputs:
                x = sigmoid_or_softmax(x, dim=1)
            return x

    def weight_decay_params(self):
        return {
            "decay_params": [self.finetune.weight],
            "other_params": [self.finetune.bias]
            + (list(self.combine.parameters()) if self.combine is not None else []),
        }


class SelectOutputs(ModuleEnhanced):
    def __init__(self, model: nn.Module, indices: torch.Tensor):
        super(SelectOutputs, self).__init__()
        self.model = model
        self.indices = indices

    def forward(self, x):
        x = self.model(x)
        assert x.dim() == 2
        x = x[:, self.indices]
        return x


class MLP(ModuleEnhanced):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        representation_state_dict: Optional[dict[str, torch.Tensor]] = None,
        finetune_scale=1.0,
        activation=nn.ReLU(),
    ):
        super(MLP, self).__init__()
        assert num_layers > 0

        layers = []
        self.activation = activation

        for layer in range(num_layers):
            if layer == 0:
                layers.append(
                    nn.Sequential(nn.Linear(input_dim, hidden_dim), activation)
                )
            else:
                layers.append(
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation)
                )

        self.representation = ModuleEnhanced.enhance(nn.Sequential(*layers))
        if representation_state_dict is not None:
            self.representation.load_state_dict(representation_state_dict)

        self.finetune = ModuleEnhanced.enhance(nn.Linear(hidden_dim, output_dim))
        self.finetune_scale = finetune_scale

    def forward(self, x: torch.Tensor):
        out = self.representation(x)
        out = self.finetune_scale * self.finetune(out)
        return out


class TruncatedMLP(MLP):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        representation_state_dict: Optional[dict[str, torch.Tensor]] = None,
        finetune_scale=1.0,
        activation=nn.ReLU(),
        truncation_factor=0.5,
    ):
        orig_input_dim = input_dim
        input_dim = int(input_dim * truncation_factor)
        truncation_factor = input_dim / orig_input_dim
        super(TruncatedMLP, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            representation_state_dict,
            finetune_scale,
            activation,
        )
        self.truncation_factor = truncation_factor

    def forward(self, x: torch.Tensor):
        x = x[:, : int(x.shape[1] * self.truncation_factor)]
        out = self.representation(x)
        out = self.finetune_scale * self.finetune(out)
        return out
