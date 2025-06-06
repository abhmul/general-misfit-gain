from typing import Callable, Generic, Literal, Sequence, TypeVar
from dataclasses import dataclass
import itertools as it

import torch

from ..params import Dataset, DEFAULT_DEVICE
from ..datasets import DataBatch, LabeledDataBatch, StrongWeakDataBatch
from ..metrics import LossFunction, Stat, Accuracy, CrossEntropy, KLDivergence
from .monte_carlo import MonteCarloEstimation
from ..torch_utils import ModuleEnhanced
from ..py_utils import peek


@dataclass
class LossSpec:
    name1: str
    name2: str
    loss_fn: LossFunction


@dataclass
class StatSpec:
    name: str
    stat_fn: Stat


@dataclass
class ModuleSpec:
    name: str
    model: ModuleEnhanced


T = TypeVar("T", bound=DataBatch)


class EstimateModelLosses(MonteCarloEstimation, Generic[T]):

    def __init__(
        self,
        dataset: Dataset,
        data_batch_type: type[T],
        apply_model_dict: dict[str, Callable[[T], torch.Tensor | None]],
        losses: list[LossSpec] | None = None,
        stats: list[StatSpec] | None = None,
        device: torch.device = DEFAULT_DEVICE,
    ):
        if losses is None:
            losses = []
        if stats is None:
            stats = []

        # Check the names in the provided dictionaries
        model_names = set(apply_model_dict.keys())
        for loss_spec in losses:
            assert loss_spec.name1 in model_names
            assert loss_spec.name2 in model_names
        for stat_spec in stats:
            assert stat_spec.name in model_names

        self.dataset = dataset
        self.data_batch_type = data_batch_type
        self.apply_model_dict = apply_model_dict
        self.losses = losses
        self.stats = stats
        self._device = device

        print(f"Using device: {self._device} for estimating losses between models")

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.no_grad()
    def estimate_batch(
        self,
        tensors: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        data_batch = self.data_batch_type.read_data_batch(tensors, device=self.device)
        model_outputs = {
            name: model_func(data_batch)
            for name, model_func in self.apply_model_dict.items()
        }

        metrics_dict: dict[str, torch.Tensor] = {}
        for loss_spec in self.losses:
            outputs1 = model_outputs[loss_spec.name1]
            outputs2 = model_outputs[loss_spec.name2]
            if outputs1 is None or outputs2 is None:
                continue

            loss = loss_spec.loss_fn(outputs1, outputs2)
            metrics_dict[
                f"{loss_spec.name1}<-{loss_spec.name2}_{loss_spec.loss_fn.__name__}"
            ] = loss
        for stat_spec in self.stats:
            outputs = model_outputs[stat_spec.name]
            if outputs is None:
                continue

            stat = stat_spec.stat_fn(outputs)
            metrics_dict[f"{stat_spec.name}_{stat_spec.stat_fn.__name__}"] = stat

        return metrics_dict


class EstimatedLabeledModelLosses(EstimateModelLosses[LabeledDataBatch]):
    GT = "gt"
    GT_LOGITS = "gt_logits"

    def __init__(
        self,
        dataset: Dataset,
        models: dict[str, ModuleEnhanced],
        losses: list[LossSpec] | None = None,
        stats: list[StatSpec] | None = None,
    ):
        assert len(models) > 0
        first_model = peek(models.values())
        device = peek(first_model.parameters()).device
        for model in models.values():
            assert device == peek(model.parameters()).device

        def create_model_func(
            model: ModuleEnhanced,
        ) -> Callable[[LabeledDataBatch], torch.Tensor]:
            return lambda data: model(data.x)

        super().__init__(
            dataset=dataset,
            data_batch_type=LabeledDataBatch,
            apply_model_dict={
                **{name: create_model_func(model) for name, model in models.items()},
                self.GT: lambda data: data.y,
                self.GT_LOGITS: lambda data: data.y_logits,
            },
            losses=losses,
            stats=stats,
            device=device,
        )


class EstimateWeakToStrong(EstimateModelLosses[StrongWeakDataBatch]):
    GT = "gt"
    GT_LOGITS = "gt_logits"
    STRONG = "strong"
    WEAK = "weak"

    def __init__(
        self,
        dataset: Dataset,
        strong_models: dict[str, ModuleEnhanced],
        weak_models: dict[str, ModuleEnhanced],
        losses: list[LossSpec] | None = None,
        stats: list[StatSpec] | None = None,
    ):
        # Check no overlap between strong and weak model names
        assert len(strong_models) > 0
        assert len(weak_models) > 0
        assert not set(strong_models.keys()) & set(weak_models.keys())
        all_models = {**strong_models, **weak_models}
        first_model = peek(all_models.values())
        device = peek(first_model.parameters()).device
        for model in all_models.values():
            assert device == peek(model.parameters()).device

        def create_model_func(
            model: ModuleEnhanced,
            model_type: str,
        ) -> Callable[[StrongWeakDataBatch], torch.Tensor]:
            if model_type == self.STRONG:
                return lambda data: model(data.strong_x)
            elif model_type == self.WEAK:
                return lambda data: model(data.weak_x)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        super().__init__(
            dataset=dataset,
            data_batch_type=StrongWeakDataBatch,
            apply_model_dict={
                **{
                    name: create_model_func(model, model_type=self.STRONG)
                    for name, model in strong_models.items()
                },
                **{
                    name: create_model_func(model, model_type=self.WEAK)
                    for name, model in weak_models.items()
                },
                self.GT: lambda data: data.y,
                self.GT_LOGITS: lambda data: data.y_logits,
            },
            losses=losses,
            stats=stats,
            device=device,
        )

    @staticmethod
    def generate_default_loss_specs(
        dual_stgt: ModuleSpec, dual_st: ModuleSpec, dual_wk: ModuleSpec
    ):
        losses_list = [
            # WRT the ground truth labels
            LossSpec(
                name1=dual_stgt.name,
                name2=EstimateWeakToStrong.GT,
                loss_fn=CrossEntropy(output_logits=True, label_logits=False),
            ),
            LossSpec(
                name1=dual_st.name,
                name2=EstimateWeakToStrong.GT,
                loss_fn=CrossEntropy(output_logits=True, label_logits=False),
            ),
            LossSpec(
                name1=dual_wk.name,
                name2=EstimateWeakToStrong.GT,
                loss_fn=CrossEntropy(output_logits=True, label_logits=False),
            ),
            LossSpec(
                name1=dual_stgt.name,
                name2=EstimateWeakToStrong.GT,
                loss_fn=Accuracy(output_logits=True, label_logits=False, hard=True),
            ),
            LossSpec(
                name1=dual_st.name,
                name2=EstimateWeakToStrong.GT,
                loss_fn=Accuracy(output_logits=True, label_logits=False, hard=True),
            ),
            LossSpec(
                name1=dual_wk.name,
                name2=EstimateWeakToStrong.GT,
                loss_fn=Accuracy(output_logits=True, label_logits=False, hard=True),
            ),
            # WRT the primal_gt model
            LossSpec(
                name1=dual_stgt.name,
                name2=dual_st.name,
                loss_fn=KLDivergence(output_logits=True, label_logits=True),
            ),
            LossSpec(
                name1=dual_stgt.name,
                name2=dual_wk.name,
                loss_fn=KLDivergence(output_logits=True, label_logits=True),
            ),
            LossSpec(
                name1=dual_stgt.name,
                name2=dual_st.name,
                loss_fn=Accuracy(output_logits=True, label_logits=True, hard=True),
            ),
            LossSpec(
                name1=dual_stgt.name,
                name2=dual_wk.name,
                loss_fn=Accuracy(output_logits=True, label_logits=True, hard=True),
            ),
            # WRT the primal_st model
            LossSpec(
                name1=dual_st.name,
                name2=dual_wk.name,
                loss_fn=KLDivergence(output_logits=True, label_logits=True),
            ),
            LossSpec(
                name1=dual_st.name,
                name2=dual_wk.name,
                loss_fn=Accuracy(output_logits=True, label_logits=True, hard=True),
            ),
            # WRT forward_kl
            LossSpec(
                name1=dual_wk.name,
                name2=dual_st.name,
                loss_fn=KLDivergence(output_logits=True, label_logits=True),
            ),
            LossSpec(
                name1=dual_wk.name,
                name2=dual_st.name,
                loss_fn=CrossEntropy(output_logits=True, label_logits=True),
            ),
        ]

        return losses_list
