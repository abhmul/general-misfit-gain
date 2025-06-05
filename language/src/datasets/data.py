from typing import Callable, Generic, Iterator, Optional, Sequence, TypeVar
from abc import ABC, abstractmethod

from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from ..params import DEFAULT_DEVICE
from ..torch_utils import ModuleEnhanced
from ..py_utils import peek

# Mimics Self type from Python >=3.11
_Self = TypeVar("_Self", bound="DataBatch")
T = TypeVar("T", bound="DataBatch")


class DataBatch(ABC):
    @classmethod
    @abstractmethod
    def read_data_batch(
        cls: type[_Self], tensors: Sequence[torch.Tensor], device: torch.device
    ) -> _Self:
        pass

    @abstractmethod
    def to(self: _Self, *args, **kwargs) -> _Self:
        pass

    @abstractmethod
    def tensors(self: _Self) -> tuple[torch.Tensor, ...]:
        pass


@dataclass
class UnlabeledDataBatch(DataBatch):
    x: torch.Tensor

    def to(self, *args, **kwargs):
        return self.__class__(self.x.to(*args, **kwargs))

    @classmethod
    def read_data_batch(
        cls: type["UnlabeledDataBatch"],
        tensors: Sequence[torch.Tensor],
        device: torch.device,
    ) -> "UnlabeledDataBatch":
        assert len(tensors) == 1
        return cls(x=tensors[0]).to(device)

    def tensors(self):
        return (self.x,)


@dataclass
class LabeledDataBatch(DataBatch):
    x: torch.Tensor
    y: torch.Tensor
    y_logits: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        return self.__class__(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            y_logits=(
                self.y_logits.to(*args, **kwargs) if self.y_logits is not None else None
            ),
        )

    # Normally I'd make this a staticmethod,
    # but making it a classmethod allows mypy
    # to understand how inheritance works.
    @classmethod
    def read_data_batch(
        cls: type["LabeledDataBatch"],
        tensors: Sequence[torch.Tensor],
        device: torch.device | None = None,
    ) -> "LabeledDataBatch":
        assert 2 <= len(tensors) <= 3
        batch_has_logits = len(tensors) == 3

        data_batch = cls(
            x=tensors[0],
            y=tensors[1],
            y_logits=(tensors[2] if batch_has_logits else None),
        )
        if device is not None:
            data_batch = data_batch.to(device)

        return data_batch

    def tensors(self):
        return (
            (self.x, self.y, self.y_logits)
            if self.y_logits is not None
            else (self.x, self.y)
        )


@dataclass
class StrongWeakDataBatch(DataBatch):
    strong_x: torch.Tensor
    weak_x: torch.Tensor
    y: torch.Tensor
    y_logits: Optional[torch.Tensor] = None

    @classmethod
    def read_data_batch(
        cls: type["StrongWeakDataBatch"],
        tensors: Sequence[torch.Tensor],
        device: torch.device,
    ) -> "StrongWeakDataBatch":
        assert 3 <= len(tensors) <= 4
        batch_has_logits = len(tensors) == 4

        data_batch = cls(
            strong_x=tensors[0],
            weak_x=tensors[1],
            y=tensors[2],
            y_logits=(tensors[3] if batch_has_logits else None),
        ).to(device)

        return data_batch

    def to(self, *args, **kwargs):
        return self.__class__(
            strong_x=self.strong_x.to(*args, **kwargs),
            weak_x=self.weak_x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            y_logits=(
                self.y_logits.to(*args, **kwargs) if self.y_logits is not None else None
            ),
        )

    def tensors(self):
        return (
            (self.strong_x, self.weak_x, self.y, self.y_logits)
            if self.y_logits is not None
            else (self.strong_x, self.weak_x, self.y)
        )


class TransformDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        transform: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
    ):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[torch.Tensor, ...]:
        # Currently only supporting these types of indexing
        assert (
            isinstance(idx, int)
            or isinstance(idx, slice)
            or isinstance(idx, torch.Tensor)
        )
        tensors: tuple[torch.Tensor, ...] = self.dataset[idx]
        assert isinstance(tensors, tuple)
        if isinstance(idx, int):
            # Add a batch dimension
            tensors = tuple(tensor.unsqueeze(0) for tensor in tensors)

        transformed_tensors = self.transform(tensors)

        if isinstance(idx, int):
            # Remove batch dimension
            transformed_tensors = tuple(
                tensor.squeeze(0) for tensor in transformed_tensors
            )

        return transformed_tensors


class ModelEmbeddingsLabeledDataset(TransformDataset):
    def __init__(
        self,
        dataset: Dataset,
        model: ModuleEnhanced,
    ):
        device = peek(model.parameters()).device
        assert all(param.device == device for param in model.parameters())

        @torch.no_grad()
        def transform(
            tensors: tuple[torch.Tensor, ...], model=model, device=device
        ) -> tuple[torch.Tensor, ...]:
            model.eval()
            data_batch = LabeledDataBatch.read_data_batch(tensors, device=device)
            transformed_batch = LabeledDataBatch(
                x=model(data_batch.x),
                y=data_batch.y,
                y_logits=data_batch.y_logits,
            )

            return transformed_batch.tensors()

        super().__init__(
            dataset,
            transform=transform,
        )

        self.model = model
        self.device = device

        print(f"Using device {device} for transforming dataset")


class StrongWeakEmbeddingsDataset(TransformDataset):
    def __init__(
        self,
        dataset: Dataset,
        st_model: ModuleEnhanced,
        wk_model: ModuleEnhanced,
    ):
        device = peek(st_model.parameters()).device
        assert all(param.device == device for param in st_model.parameters())
        assert all(param.device == device for param in wk_model.parameters())

        @torch.no_grad()
        def transform(
            tensors: tuple[torch.Tensor, ...], st_model=st_model, wk_model=wk_model
        ) -> tuple[torch.Tensor, ...]:
            st_model.eval()
            wk_model.eval()
            data_batch = StrongWeakDataBatch.read_data_batch(tensors, device=device)
            transformed_data_batch = StrongWeakDataBatch(
                strong_x=st_model(data_batch.strong_x),
                weak_x=wk_model(data_batch.weak_x),
                y=data_batch.y,
                y_logits=data_batch.y_logits,
            )
            return transformed_data_batch.tensors()

        super().__init__(
            dataset,
            transform=transform,
        )

        self.st_model = st_model
        self.wk_model = wk_model
        self.device = device

        print(f"Using device {device} for transforming dataset")
