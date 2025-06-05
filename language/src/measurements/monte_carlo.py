from abc import ABC, abstractmethod
from typing import Collection, Dict, Optional, Protocol, Sequence
import math
from collections import defaultdict

from tqdm import tqdm  # type: ignore[import-untyped]
import torch
import torch.utils.data as data

from ..params import EP, DEFAULT_DEVICE, Dataset, CPU_GENERATOR
from ..py_utils import Number


class ResultsListener(Protocol):
    def update(self, results: dict[str, Number]): ...


class MonteCarloEstimation(ABC):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @abstractmethod
    def estimate_batch(
        self, tensors: Sequence[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        pass

    def estimate(
        self,
        batch_size: int,
        listeners: Optional[Collection[ResultsListener]] = None,
    ) -> Dict[str, Number]:
        estimator = MonteCarloEstimator(device=self.device)

        data_loader = data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=CPU_GENERATOR,
        )

        pbar = tqdm(enumerate(data_loader), total=len(data_loader))

        for i, tensors in pbar:
            metrics = self.estimate_batch(tensors)
            estimator.update(metrics)

            results = estimator.results()
            if listeners is not None:
                for listener in listeners:
                    listener.update(results)

            # Update the progress bar
            pbar.set_postfix(
                {
                    f"{name}": estimator.running_means[name].item()
                    for name in metrics.keys()
                }
            )
            pbar.refresh()

        return estimator.results()


class MonteCarloEstimator:

    def __init__(self, device=DEFAULT_DEVICE) -> None:
        self.device = device
        self.float_type = torch.float32
        self.running_means: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.tensor(0.0, device=device, dtype=self.float_type)
        )
        self.running_sqmeans: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.tensor(0.0, device=device, dtype=self.float_type)
        )
        self.running_counts: Dict[str, int] = defaultdict(int)

    def get_variance(self, name: str) -> torch.Tensor:
        counts = self.running_counts[name]
        if counts <= 1:
            res = torch.tensor(
                0.0,
                device=self.device,
                dtype=self.float_type,
            )
        else:
            res = (
                (self.running_sqmeans[name] - self.running_means[name] ** 2)
                * counts
                / (counts - 1)
            )
        if res < 0:
            print(f"variance became negative! {res}")
            res = torch.tensor(
                0.0,
                device=self.device,
                dtype=self.float_type,
            )
        return res

    def get_error(self, name: str) -> torch.Tensor:
        return (
            2
            * torch.sqrt(self.get_variance(name))
            / math.sqrt(self.running_counts[name])
        )

    def update(
        self,
        metrics: Dict[str, torch.Tensor],
    ) -> None:
        for k, v in metrics.items():
            old_count = self.running_counts[k]
            self.running_counts[k] += v.shape[0]

            self.running_means[k] = (
                self.running_means[k] * old_count / self.running_counts[k]
                + torch.sum(v, dtype=self.float_type) / self.running_counts[k]
            )
            self.running_sqmeans[k] = (
                self.running_sqmeans[k] * old_count / self.running_counts[k]
                + torch.sum(v**2, dtype=self.float_type) / self.running_counts[k]
            )

    def results(self) -> dict[str, Number]:
        means = {
            f"{name}__mean": val.item() for name, val in self.running_means.items()
        }
        stds = {
            f"{name}__std": torch.sqrt(self.get_variance(name)).item()
            for name in self.running_sqmeans.keys()
        }
        err = {
            f"{name}__err": self.get_error(name).item()
            for name in self.running_counts.keys()
        }
        return {
            **means,
            **stds,
            **err,
        }

    def estimated_needed_samples(self, name: str, tolerance: float) -> Number:
        return (
            4 * self.get_variance(name) / (tolerance * self.running_means[name]) ** 2
        ).item()
