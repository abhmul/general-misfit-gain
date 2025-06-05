from typing import Iterator, Optional, TypeVar, Generic, Sequence
from abc import ABC, abstractmethod

import math
import torch
import torch.nn as nn
import torch.utils.data as data

from ..params import get_generator, Dataset, CPU_GENERATOR
from ..py_utils import SizedGenerator


class DataSampler(ABC):
    @abstractmethod
    def sample(self, batch_size: int) -> Sequence[torch.Tensor]:
        pass

    def as_generator(
        self, batch_size: int, num_batches: Optional[int] = None
    ) -> Iterator[Sequence[torch.Tensor]]:
        if num_batches is None:
            while True:
                yield self.sample(batch_size)
        else:
            for _ in range(num_batches):
                yield self.sample(batch_size)

    def as_finite_generator(
        self, batch_size: int, num_batches: int
    ) -> SizedGenerator[Sequence[torch.Tensor]]:
        return SizedGenerator(self.as_generator(batch_size, num_batches), num_batches)


class SyntheticDataSampler(DataSampler):
    @property
    @abstractmethod
    def model(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def sample_input_data(self, batch_size: int) -> torch.Tensor:
        pass

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x = self.sample_input_data(batch_size)
            y_logit = self.model(x)
            if self.output_dim == 1:
                y = torch.sigmoid(y_logit)
            else:
                y = torch.softmax(y_logit, dim=-1)
            return x, y, y_logit


class SyntheticNormalDataSampler(SyntheticDataSampler):
    def __init__(
        self,
        model: nn.Module,
        input_dim: int,
        output_dim: int,
        var: float,
    ):
        self._model = model
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.var = var
        self.std = math.sqrt(var)
        self.device = next(model.parameters()).device

        print(f"Using model device for synthetic data generation: {self.device}")

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def sample_input_data(self, batch_size: int) -> torch.Tensor:
        return torch.normal(
            0,
            self.std,
            (batch_size, self.input_dim),
            generator=get_generator(self.device),
            device=self.device,
        )


class SyntheticDatasetDataSampler(SyntheticDataSampler):
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        input_dim: int,
        output_dim: int,
    ):
        self._model = model
        self.dataset = dataset
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.device = next(model.parameters()).device

        print(f"Using model device for synthetic data generation: {self.device}")

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def sample_input_data(self, batch_size: int) -> torch.Tensor:
        inds = torch.randint(
            len(self.dataset),
            size=(batch_size,),
            device=self.device,
            generator=get_generator(self.device),
        )
        # Assumes input data is the first element of the dataset
        x_data = torch.stack([self.dataset[idx][0] for idx in inds], dim=0)
        return x_data


class DatasetSampler(DataSampler):
    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=CPU_GENERATOR,
        )

        self.data_iterator = iter(self.loader)

    def sample(self, batch_size: int) -> Sequence[torch.Tensor]:
        assert batch_size == self.loader.batch_size  # gross but wtv
        return next(self.data_iterator)
