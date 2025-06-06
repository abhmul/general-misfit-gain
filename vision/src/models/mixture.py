from abc import ABC, abstractmethod
from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..params import CPU_GENERATOR
from ..torch_utils import (
    rand_unit_vectors,
    hardmax,
    ModuleEnhanced,
    project_to_simplex,
    clamped_log,
    combinations,
    bitarray2dec,
)


class HomogenousFeatures(ModuleEnhanced, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def num_states(self) -> int:
        pass

    def prepend(self, model: ModuleEnhanced, input_dim: int, output_dim: int):
        assert output_dim == self.input_dim
        return SequentialHomogenousFeatures(
            [model, self],
            input_dim=input_dim,
            output_dim=self.output_dim,
            num_states=self.num_states,
        )


class SequentialHomogenousFeatures(HomogenousFeatures):
    def __init__(
        self,
        transforms: list[ModuleEnhanced],
        input_dim: int,
        output_dim: int,
        num_states: int,
    ):
        super(SequentialHomogenousFeatures, self).__init__()
        self.transforms = nn.Sequential(*transforms)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._num_states = num_states

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_states(self) -> int:
        return self._num_states

    def forward(self, x):
        return self.transforms(x)

    def prepend(self, model, input_dim, output_dim):
        assert output_dim == self.input_dim
        return SequentialHomogenousFeatures(
            [model] + list(self.transforms),
            input_dim=input_dim,
            output_dim=self.output_dim,
            num_states=self.num_states,
        )

    def project_parameters(self):
        for module in self.transforms:
            assert isinstance(module, ModuleEnhanced)
            module.project_parameters()


class PrimalThreshold(HomogenousFeatures):
    def __init__(self, thresholds: torch.Tensor, num_combinations: int):
        super(PrimalThreshold, self).__init__()
        assert thresholds.dim() == 1
        self.thresholds = thresholds
        self.num_combinations = num_combinations

    @property
    def input_dim(self) -> int:
        return self.thresholds.shape[0]

    @property
    def num_states(self) -> int:
        return 2**self.num_combinations

    @property
    def output_dim(self) -> int:
        return comb(self.input_dim, self.num_combinations)

    @staticmethod
    def functional_apply(x, thresholds, num_combinations):
        assert x.dim() == 2
        assert x.shape[1] == thresholds.shape[0]
        dtype = x.dtype

        active = x > thresholds[None, :]
        # Choose all subsets from active of size self.combinations
        x = combinations(active, r=num_combinations, dim=1)
        x = bitarray2dec(x, dim=2, dtype=torch.long, device=x.device)
        x = F.one_hot(x, num_classes=2**num_combinations).to(dtype=dtype)

        return x

    def forward(self, x):
        self.functional_apply(x, self.thresholds, self.num_combinations)


class MeanThreshold(HomogenousFeatures):
    def __init__(self, input_dim: int, num_combinations: int):
        super(MeanThreshold, self).__init__()
        self._input_dim = input_dim
        self.num_combinations = num_combinations

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def num_states(self) -> int:
        return 2**self.num_combinations

    @property
    def output_dim(self) -> int:
        return comb(self.input_dim, self.num_combinations)

    def forward(self, x):
        thresholds = torch.mean(x, dim=0)
        return PrimalThreshold.functional_apply(x, thresholds, self.num_combinations)


class BagOfDecisionBoundaries(HomogenousFeatures):
    def __init__(
        self, input_dim: int, output_dim: int, num_states: int = 2, soft=False
    ):
        super(BagOfDecisionBoundaries, self).__init__()

        unit_vectors = rand_unit_vectors(
            output_dim * num_states, input_dim + 1, generator=CPU_GENERATOR
        ).reshape(output_dim, num_states, input_dim + 1)
        self.decision_boundaries = nn.Parameter(
            unit_vectors[:, :, :-1], requires_grad=soft
        )
        self.biases = nn.Parameter(unit_vectors[:, :, -1], requires_grad=soft)
        self._num_states = num_states
        if soft:
            raise NotImplementedError

    @property
    def input_dim(self) -> int:
        return self.decision_boundaries.shape[2]

    @property
    def output_dim(self) -> int:
        return self.decision_boundaries.shape[0]

    @property
    def num_states(self) -> int:
        return self._num_states

    def forward(self, x):
        assert x.dim() == 2
        assert x.shape[1] == self.input_dim

        x = torch.einsum("bi,osi->bos", x, self.decision_boundaries)
        x += self.biases[None]
        x = hardmax(x, dim=2)

        return x


class SparseBagOfDecisionBoundaries(BagOfDecisionBoundaries):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_states: int = 2,
        weight: int = 50,
        soft: bool = False,
    ):
        super(SparseBagOfDecisionBoundaries, self).__init__(
            input_dim=weight, output_dim=output_dim, num_states=num_states, soft=soft
        )

        self.weight = weight
        self._input_dim = input_dim

        self.selection = nn.Parameter(
            torch.randint(
                input_dim,
                size=(output_dim, weight),
                generator=CPU_GENERATOR,
            ),
            requires_grad=False,
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2
        assert x.shape[1] == self.input_dim
        batch_size = x.shape[0]

        # Select the indices in self.selection[i, :] for each i=0...output_dim-1
        # Should result in a tensor of shape (batch_size, output_dim, weight)
        x = x[:, self.selection]
        # assert x.shape == (batch_size, self.output_dim, self.weight)

        x = torch.einsum("bow,osw->bos", x, self.decision_boundaries)
        x += self.biases[None]
        x = hardmax(x, dim=2)

        return x


class DualMixtureLayer(ModuleEnhanced):
    def __init__(self, input_dim: int, output_dim: int, num_states: int = 2):
        super(DualMixtureLayer, self).__init__()
        self.num_states = num_states
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.mixture_weights = nn.Parameter(torch.zeros((self.input_dim,)))
        self.conditional_weights = nn.Parameter(
            torch.zeros((self.output_dim, self.input_dim, self.num_states))
        )

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1:] == (self.input_dim, self.num_states)

        mixture_probs = torch.softmax(self.mixture_weights, dim=0)
        conditional_probs = torch.softmax(self.conditional_weights, dim=0)
        x = torch.einsum("bis,i,kis->bk", x, mixture_probs, conditional_probs)
        # x = torch.einsum("bis,kis->bk", x, conditional_probs)
        return x

    def to_primal(self):
        mixture_layer = PrimalMixtureLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_states=self.num_states,
        )
        with torch.no_grad():
            mixture_layer.mixture_probs.data = torch.softmax(
                self.mixture_weights.data.clone(), dim=0
            )
            mixture_layer.conditional_probs.data = torch.softmax(
                self.conditional_weights.data.clone(), dim=0
            )
        return mixture_layer


class PrimalMixtureLayer(ModuleEnhanced):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_states: int = 2,
    ):
        super(PrimalMixtureLayer, self).__init__()

        self.num_states = num_states
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.mixture_probs = nn.Parameter(
            torch.full((self.input_dim,), 1 / self.input_dim)
        )
        self.conditional_probs = nn.Parameter(
            torch.full(
                (self.output_dim, self.input_dim, self.num_states), 1 / self.output_dim
            )
        )

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1:] == (self.input_dim, self.num_states)

        x = torch.einsum("bis,i,kis->bk", x, self.mixture_probs, self.conditional_probs)
        return x

    def project_parameters(self):
        self.mixture_probs.data = project_to_simplex(self.mixture_probs.data, dim=0)
        self.conditional_probs.data = project_to_simplex(
            self.conditional_probs.data, dim=0
        )

    def to_dual(self):
        mixture_layer = DualMixtureLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_states=self.num_states,
        )
        mixture_layer.mixture_weights.data = clamped_log(
            self.mixture_probs.data.clone()
        )
        mixture_layer.conditional_weights.data = clamped_log(
            self.conditional_probs.data.clone()
        )
        return mixture_layer


class HomogenousMixtureModel(ModuleEnhanced):
    def __init__(
        self,
        features_model: HomogenousFeatures,
        output_dim: int,
        no_features_grad: bool = True,
        use_dual_weights: bool = False,
    ):
        super(HomogenousMixtureModel, self).__init__()
        self.features_model = features_model
        self.output_dim = output_dim
        self.use_dual_weights = use_dual_weights
        if use_dual_weights:
            self.mixture_layer: DualMixtureLayer | PrimalMixtureLayer = (
                DualMixtureLayer(
                    input_dim=features_model.output_dim,
                    output_dim=output_dim,
                    num_states=features_model.num_states,
                )
            )
        else:
            self.mixture_layer = PrimalMixtureLayer(
                input_dim=features_model.output_dim,
                output_dim=output_dim,
                num_states=features_model.num_states,
            )

        self.no_features_grad = no_features_grad

    @property
    def input_dim(self) -> int:
        return self.features_model.input_dim

    @property
    def num_states(self) -> int:
        return self.features_model.num_states

    def forward(self, x):
        if self.no_features_grad:
            with torch.no_grad():
                x = self.features_model(x)
        else:
            x = self.features_model(x)

        x = self.mixture_layer(x)
        return x

    def project_parameters(self):
        self.mixture_layer.project_parameters()
        self.features_model.project_parameters()
