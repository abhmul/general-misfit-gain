from typing import Iterator, Optional, Sequence
from dataclasses import dataclass

from collections import deque
import torch.utils
from tqdm import tqdm, trange  # type: ignore[import-untyped]

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.utils.data as data

from ..torch_utils import ModuleEnhanced
from ..py_utils import Number
from ..params import Dataset, CPU_GENERATOR, TMP_DIR
from ..file_utils import safe_open_file
from ..datasets import DataSampler, LabeledDataBatch
from ..metrics.losses import ClassificationLossFunction
from ..measurements import MonteCarloEstimator


@dataclass
class TrainStepResult:
    loss: torch.Tensor
    step_norm: torch.Tensor
    grad_norm: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass
class EvaluateStepResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class ClassificationTrainer:

    def __init__(
        self,
        model: ModuleEnhanced,
        optimizer: Optimizer,
        loss_fn: ClassificationLossFunction,
        metrics: list[ClassificationLossFunction] | None = None,
        scheduler: Optional[LRScheduler] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics if metrics is not None else []
        self.scheduler = scheduler

        self.device = next(model.parameters()).device
        print(f"Using model device for training: {self.device}")

    def train_step(self, data_batch: LabeledDataBatch) -> TrainStepResult:
        """
        Assumes model is already in train mode if desired
        """

        # Convert data bach to device
        data_batch = data_batch.to(self.device)

        self.optimizer.zero_grad()
        out = self.model(data_batch.x)
        if self.loss_fn.label_logits:
            assert data_batch.y_logits is not None
            loss = self.loss_fn(out, data_batch.y_logits)
        else:
            loss = self.loss_fn(out, data_batch.y)
        loss.mean().backward()

        # grad_norm = torch.norm(
        #     torch.cat([p.grad.flatten() for p_group in self.optimizer.param_groups for p in p_group["params"]]), p=2  # type: ignore[union-attr]
        # )
        with torch.no_grad():
            metrics: dict[str, torch.Tensor] = {}
            if self.metrics is not None:
                for metric in self.metrics:
                    if metric.label_logits:
                        assert data_batch.y_logits is not None
                        y = data_batch.y_logits
                    else:
                        y = data_batch.y
                    metrics[metric.__name__] = metric(out, y)

        prev_params = [
            [p.clone() for p in p_group["params"]]
            for p_group in self.optimizer.param_groups
        ]
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.model.project_parameters()
        scaled_step_norm = torch.norm(
            torch.cat(
                [
                    (p - prev_p).flatten()
                    for p_group, prev_p_group in zip(
                        self.optimizer.param_groups, prev_params
                    )
                    for p, prev_p in zip(p_group["params"], prev_p_group)
                ]
            ),
            p=2,
        )
        grad_norm = torch.norm(
            torch.cat([p.grad.flatten() for p_group in self.optimizer.param_groups for p in p_group["params"]]), p=2  # type: ignore[union-attr]
        )

        return TrainStepResult(
            loss=loss, step_norm=scaled_step_norm, grad_norm=grad_norm, metrics=metrics
        )

    @torch.no_grad()
    def evaluate_step(self, data_batch: LabeledDataBatch) -> EvaluateStepResult:
        """
        Assumes model is already in eval mode if desired
        """

        # Convert data bach to device
        data_batch = data_batch.to(self.device)

        out = self.model(data_batch.x)
        if self.loss_fn.label_logits:
            assert data_batch.y_logits is not None
            loss = self.loss_fn(out, data_batch.y_logits)
        else:
            loss = self.loss_fn(out, data_batch.y)

        metrics: dict[str, torch.Tensor] = {}
        for metric in self.metrics:
            if metric.label_logits:
                assert data_batch.y_logits is not None
                y = data_batch.y_logits
            else:
                y = data_batch.y
            metrics[metric.__name__] = metric(out, y)

        return EvaluateStepResult(loss=loss, metrics=metrics)


class RunningAverage:
    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size
        self.running_sum = 0.0
        self.count = 0
        if window_size is not None:
            self.values_buffer: Optional[deque[float]] = deque(maxlen=window_size)
        else:
            self.values_buffer = None

    def add(self, value: float):
        self.count += 1
        if self.window_size is not None:
            assert self.count <= self.window_size + 1
            assert self.values_buffer is not None
            if self.count == self.window_size + 1:
                assert len(self.values_buffer) == self.window_size
                self.running_sum -= self.values_buffer.pop()
                self.count = self.window_size
            self.values_buffer.appendleft(value)

        self.running_sum += value

    def mean(self) -> float:
        return self.running_sum / self.count


class SamplingClassificationTrainer(ClassificationTrainer):
    def __init__(
        self,
        model: ModuleEnhanced,
        optimizer: Optimizer,
        loss_fn: ClassificationLossFunction,
        data_sampler: DataSampler,
        metrics: list[ClassificationLossFunction] | None = None,
    ):
        super(SamplingClassificationTrainer, self).__init__(
            model,
            optimizer,
            loss_fn,
            metrics=metrics,
        )
        self.data_sampler = data_sampler

    def train(
        self,
        num_samples: int,
        batch_size: int,
        average_window: int = 10,
        update_pbar_every: int = 10,
    ):
        assert average_window > 0
        ra = RunningAverage(average_window)
        update_norm_ra = RunningAverage(average_window)
        grad_norm_ra = RunningAverage(average_window)
        metrics_ra = {
            metric.__name__: RunningAverage(average_window) for metric in self.metrics
        }

        pbar = trange(0, num_samples, batch_size)
        postfix: dict[str, Number] = {}
        pbar.set_postfix(postfix)
        for i, start in enumerate(pbar):
            this_batch_size = min(batch_size, num_samples - start)
            tensors = self.data_sampler.sample(this_batch_size)
            data_batch = LabeledDataBatch.read_data_batch(tensors, device=self.device)
            result = self.train_step(data_batch)

            ra.add(result.loss.mean().item())
            update_norm_ra.add(result.step_norm.mean().item())
            grad_norm_ra.add(result.grad_norm.mean().item())
            for metric_name, metric_value in result.metrics.items():
                metrics_ra[metric_name].add(metric_value.mean().item())

            if i % update_pbar_every == 0:
                postfix["loss"] = ra.mean()
                postfix["up_norm"] = update_norm_ra.mean()
                postfix["grad_norm"] = grad_norm_ra.mean()
                for metric_name, metric_ra in metrics_ra.items():
                    postfix[metric_name] = metric_ra.mean()
                pbar.set_postfix(postfix)
                pbar.refresh()


class DatasetClassificationTrainer(ClassificationTrainer):
    def __init__(
        self,
        model: ModuleEnhanced,
        optimizer: Optimizer,
        loss_fn: ClassificationLossFunction,
        dataset: Dataset,
        metrics: list[ClassificationLossFunction] | None = None,
        val_dataset: Optional[Dataset] = None,
        scheduler: Optional[LRScheduler] = None,
    ):
        super(DatasetClassificationTrainer, self).__init__(
            model,
            optimizer,
            loss_fn,
            metrics=metrics,
            scheduler=scheduler,
        )
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.device = next(model.parameters()).device
        print(f"Using model device for training: {self.device}")

        self.validation_tmp_path = TMP_DIR / "validation.pt"

    def n_iter_per_epoch(self, batch_size: int) -> int:
        return len(range(0, len(self.dataset), batch_size))

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        average_window: Optional[int] = 10,
        update_pbar_every: int = 10,
    ):
        best_val_loss = float("inf")
        best_model_info = {}

        train_data_loader = data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=CPU_GENERATOR,
        )
        if self.val_dataset is not None:
            val_data_loader = data.DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                generator=CPU_GENERATOR,
            )
        pbar = trange(num_epochs)
        postfix: dict[str, Number] = {}
        pbar.set_postfix(postfix)
        for ep in pbar:
            self.model.train()
            ra = RunningAverage(average_window)
            update_norm_ra = RunningAverage(average_window)
            grad_norm_ra = RunningAverage(average_window)
            metrics_ra = {
                metric.__name__: RunningAverage(average_window)
                for metric in self.metrics
            }

            for i, tensors in enumerate(train_data_loader):
                data_batch = LabeledDataBatch.read_data_batch(
                    tensors, device=self.device
                )
                result = self.train_step(data_batch)
                ra.add(result.loss.mean().item())
                update_norm_ra.add(result.step_norm.mean().item())
                grad_norm_ra.add(result.grad_norm.mean().item())
                for metric_name, metric_value in result.metrics.items():
                    metrics_ra[metric_name].add(metric_value.mean().item())

                if i % update_pbar_every == 0:
                    postfix["loss"] = ra.mean()
                    postfix["up_norm"] = update_norm_ra.mean()
                    postfix["grad_norm"] = grad_norm_ra.mean()
                    for metric_name, metric_ra in metrics_ra.items():
                        postfix[metric_name] = metric_ra.mean()
                    pbar.set_postfix(postfix)
                    pbar.refresh()

            if self.val_dataset is not None:
                self.model.eval()
                ra = RunningAverage(window_size=None)
                metrics_ra = {
                    metric.__name__: RunningAverage(window_size=None)
                    for metric in self.metrics
                }
                for tensors in val_data_loader:
                    data_batch = LabeledDataBatch.read_data_batch(
                        tensors, device=self.device
                    )
                    val_result = self.evaluate_step(data_batch)
                    ra.add(val_result.loss.mean().item())
                    for metric_name, metric_value in val_result.metrics.items():
                        metrics_ra[metric_name].add(metric_value.mean().item())

                val_loss = ra.mean()

                # Save the model if it is the best so far.
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        self.model.state_dict(),
                        safe_open_file(self.validation_tmp_path),
                    )
                    best_model_info = {
                        "epoch": ep,
                        "val_loss": val_loss,
                        **{
                            f"val_{metric_name}": metric_ra.mean()
                            for metric_name, metric_ra in metrics_ra.items()
                        },
                    }
                postfix.update(
                    {
                        "val_loss": val_loss,
                        **{
                            f"best_{name}": value
                            for name, value in best_model_info.items()
                        },
                    }
                )
                pbar.set_postfix(postfix)
                pbar.refresh()

        if self.val_dataset is not None:
            self.model.load_state_dict(
                torch.load(self.validation_tmp_path, weights_only=True)
            )
            self.validation_tmp_path.unlink()
            print(
                f"Best model found at epoch {best_model_info['epoch']} with val loss {best_model_info['val_loss']}"
            )
