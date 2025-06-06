from typing import Callable, cast, Generic, TypeVar
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from ..torch_utils import clamped_log, clamped_logit


# LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LossFunction(ABC):
    @property
    @abstractmethod
    def __name__(self) -> str:
        pass

    @abstractmethod
    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass

    def reverse_loss(self) -> "ReversedLossFunction":
        return ReversedLossFunction(self)


T = TypeVar("T", bound=LossFunction)


class ReversedLossFunction(Generic[T], LossFunction):
    def __init__(self, loss_fn: T):
        self.loss_fn = loss_fn

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(labels, outputs)

    @property
    def __name__(self) -> str:
        return f"reversed_{self.loss_fn.__name__}"


class BinaryOrMultiLoss(LossFunction):
    def __init__(
        self, binary: LossFunction, multi: LossFunction, name: str | None = None
    ):
        self.binary = binary
        self.multi = multi
        self._name = f"{binary.__name__}_or_{multi.__name__}" if name is None else name

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert outputs.shape == labels.shape
        assert outputs.ndim == labels.ndim == 2

        if labels.shape[1] == 1:
            return self.binary(outputs, labels)
        else:
            return self.multi(outputs, labels)

    @property
    def __name__(self) -> str:
        return self._name


class ClassificationLossFunction(LossFunction):

    def __init__(self, output_logits: bool = False, label_logits: bool = False):
        """
        Initializes the ClassificationLossFunction.

        :param output_logits: If True, indicates that the output tensor is in logits format (raw scores).
        :param label_logits: If True, indicates that the label tensor is in logits format (raw scores).
        """
        self._output_logits = output_logits
        self._label_logits = label_logits

    @property
    def output_logits(self) -> bool:
        """
        Returns whether the outputs are logits.
        This property should be implemented by subclasses to indicate
        if the output tensor is in logits format (raw scores) or probabilities.
        """
        return self._output_logits

    @property
    def label_logits(self) -> bool:
        """
        Returns whether the labels are in logits format.
        This property should be implemented by subclasses to indicate
        if the label tensor is in logits format (raw scores) or probabilities.
        """
        return self._label_logits

    def reverse_loss(self) -> "ReversedClassificationLossFunction":
        return ReversedClassificationLossFunction(self)


S = TypeVar("S", bound=ClassificationLossFunction)


class ReversedClassificationLossFunction(
    ReversedLossFunction[S], ClassificationLossFunction
):
    def __init__(self, loss_fn: S):
        self.loss_fn = loss_fn

    @property
    def output_logits(self) -> bool:
        """
        Returns whether the outputs are logits.
        This property is inherited from the original loss function.
        """
        return self.loss_fn.label_logits

    @property
    def label_logits(self) -> bool:
        """
        Returns whether the labels are in logits format.
        This property is inherited from the original loss function.
        """
        return self.loss_fn.output_logits


class MultiCrossEntropy(ClassificationLossFunction):
    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert outputs.shape == labels.shape
        assert outputs.ndim == labels.ndim == 2

        if self.label_logits:
            labels = torch.softmax(labels, dim=1)

        if self.output_logits:
            return F.cross_entropy(outputs, labels, reduction="none")
        else:
            outputs = clamped_log(outputs)
            return F.cross_entropy(outputs, labels, reduction="none")

    @property
    def __name__(self) -> str:
        return "multi_cross_entropy"


class BinaryCrossEntropy(ClassificationLossFunction):
    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert outputs.shape == labels.shape
        assert outputs.ndim == labels.ndim == 2

        if self.label_logits:
            labels = torch.sigmoid(labels)

        if self.output_logits:
            return F.binary_cross_entropy_with_logits(
                outputs, labels, reduction="none"
            ).squeeze(1)
        else:
            return F.binary_cross_entropy(outputs, labels, reduction="none").squeeze(1)

    @property
    def __name__(self) -> str:
        return "binary_cross_entropy"


class CrossEntropy(BinaryOrMultiLoss, ClassificationLossFunction):
    def __init__(self, output_logits: bool = False, label_logits: bool = False):
        BinaryOrMultiLoss.__init__(
            self,
            BinaryCrossEntropy(output_logits, label_logits),
            MultiCrossEntropy(output_logits, label_logits),
            name="cross_entropy",
        )
        ClassificationLossFunction.__init__(self, output_logits, label_logits)


class BregmanDivergence(LossFunction):
    def __init__(self, outputs_dual: bool = False, targets_dual: bool = False):
        self.outputs_dual = outputs_dual
        self.targets_dual = targets_dual

    @abstractmethod
    def primal_potential(self, primal_values: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def dual_potential(self, dual_values: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def primal_to_dual(self, primal_values: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def dual_to_primal(self, dual_values: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert outputs.shape == targets.shape
        assert outputs.ndim == targets.ndim == 2

        if self.outputs_dual:
            outputs = self.dual_to_primal(outputs)
        if not self.targets_dual:
            targets = self.primal_to_dual(targets)

        return (
            self.primal_potential(outputs)
            + self.dual_potential(targets)
            - torch.sum(outputs * targets, dim=1)
        )

    def dual_divergence(self) -> "BregmanDivergence":
        return DualDivergence(
            self, outputs_dual=not self.targets_dual, targets_dual=not self.outputs_dual
        )


class DualDivergence(BregmanDivergence):
    def __init__(
        self,
        primal_divergence: BregmanDivergence,
        outputs_dual=False,
        targets_dual=False,
    ):
        super().__init__(outputs_dual, targets_dual)
        self.primal_divergence = primal_divergence

    def primal_potential(self, primal_values: torch.Tensor) -> torch.Tensor:
        return self.primal_divergence.dual_potential(primal_values)

    def dual_potential(self, dual_values: torch.Tensor) -> torch.Tensor:
        return self.primal_divergence.primal_potential(dual_values)

    def primal_to_dual(self, primal_values: torch.Tensor) -> torch.Tensor:
        return self.primal_divergence.dual_to_primal(primal_values)

    def dual_to_primal(self, dual_values: torch.Tensor) -> torch.Tensor:
        return self.primal_divergence.primal_to_dual(dual_values)

    @property
    def __name__(self) -> str:
        return f"dual_{self.primal_divergence.__name__}"

    def dual_divergence(self) -> "BregmanDivergence":
        return self.primal_divergence


class BinaryOrMultiBregmanDivergence(BinaryOrMultiLoss, BregmanDivergence):
    def __init__(
        self,
        binary: BregmanDivergence,
        multi: BregmanDivergence,
        name: str | None = None,
    ):
        super().__init__(binary, multi, name=name)
        self.binary = cast(BregmanDivergence, self.binary)
        self.multi = cast(BregmanDivergence, self.multi)

    def primal_potential(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.shape[1] == 1:
            return self.binary.primal_potential(probs)  # type: ignore[attr-defined]
        else:
            return self.multi.primal_potential(probs)  # type: ignore[attr-defined]

    def dual_potential(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            return self.binary.dual_potential(logits)  # type: ignore[attr-defined]
        else:
            return self.multi.dual_potential(logits)  # type: ignore[attr-defined]

    def primal_to_dual(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.shape[1] == 1:
            return self.binary.primal_to_dual(probs)  # type: ignore[attr-defined]
        else:
            return self.multi.primal_to_dual(probs)  # type: ignore[attr-defined]

    def dual_to_primal(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            return self.binary.dual_to_primal(logits)  # type: ignore[attr-defined]
        else:
            return self.multi.dual_to_primal(logits)  # type: ignore[attr-defined]


class MultiKLDivergence(BregmanDivergence, ClassificationLossFunction):
    # Output = arg1 in KL Div, Label = arg2 in KL Div
    def __init__(self, output_logits: bool = False, label_logits: bool = False):
        BregmanDivergence.__init__(
            self, outputs_dual=output_logits, targets_dual=label_logits
        )
        ClassificationLossFunction.__init__(self, output_logits, label_logits)

    def primal_potential(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(probs * clamped_log(probs), dim=1)

    def dual_potential(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(logits, dim=1)

    def primal_to_dual(self, probs: torch.Tensor) -> torch.Tensor:
        return clamped_log(probs)

    def dual_to_primal(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=1)

    @property
    def __name__(self) -> str:
        return "multi_kl_divergence"


class BinaryKLDivergence(BregmanDivergence, ClassificationLossFunction):
    # Output = arg1 in KL Div, Label = arg2 in KL Div
    def __init__(self, output_logits: bool = False, label_logits: bool = False):
        BregmanDivergence.__init__(
            self, outputs_dual=output_logits, targets_dual=label_logits
        )
        ClassificationLossFunction.__init__(self, output_logits, label_logits)

    def primal_potential(self, probs: torch.Tensor) -> torch.Tensor:
        return (
            probs * clamped_log(probs) + (1 - probs) * clamped_log(1 - probs)
        ).squeeze(1)

    def dual_potential(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.log(1 + torch.exp(logits)).squeeze(1)

    def primal_to_dual(self, probs: torch.Tensor) -> torch.Tensor:
        return clamped_logit(probs)

    def dual_to_primal(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    @property
    def __name__(self) -> str:
        return "binary_kl_divergence"


class KLDivergence(BinaryOrMultiBregmanDivergence, ClassificationLossFunction):
    def __init__(self, output_logits: bool = False, label_logits: bool = False):
        BinaryOrMultiBregmanDivergence.__init__(
            self,
            BinaryKLDivergence(output_logits, label_logits),
            MultiKLDivergence(output_logits, label_logits),
            name="kl_divergence",
        )
        ClassificationLossFunction.__init__(self, output_logits, label_logits)


class MultiAccuracy(ClassificationLossFunction):
    def __init__(
        self,
        output_logits: bool = False,
        label_logits: bool = False,
        hard: bool = False,
    ):
        super().__init__(output_logits, label_logits)
        self.hard = hard

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert outputs.shape == labels.shape
        assert outputs.ndim == labels.ndim == 2

        if self.hard:
            labels = F.one_hot(labels.argmax(dim=1), num_classes=labels.shape[1]).to(
                outputs.dtype
            )
        else:
            if self.label_logits:
                labels = torch.softmax(labels, dim=1)

        # Don't need to do anything if outputs are logits.
        preds = F.one_hot(outputs.argmax(dim=1), num_classes=outputs.shape[1]).to(
            outputs.dtype
        )

        return torch.sum(preds * labels, dim=1)

    @property
    def __name__(self) -> str:
        return "multi_accuracy"


class BinaryAccuracy(ClassificationLossFunction):
    def __init__(
        self,
        output_logits: bool = False,
        label_logits: bool = False,
        hard: bool = False,
    ):
        super().__init__(output_logits, label_logits)
        self.hard = hard

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert outputs.shape == labels.shape
        assert outputs.shape[1] == 1
        assert outputs.ndim == labels.ndim == 2

        if self.label_logits:
            if self.hard:
                labels = (labels >= 0).to(outputs.dtype)
            else:
                labels = torch.sigmoid(labels)
        else:
            if self.hard:
                labels = (labels >= 0.5).to(outputs.dtype)

        if self.output_logits:
            preds = (outputs >= 0).to(outputs.dtype)
        else:
            preds = (torch.sigmoid(outputs) >= 0.5).to(outputs.dtype)

        return (preds * labels + (1 - preds) * (1 - labels)).squeeze(1)

    @property
    def __name__(self) -> str:
        return "binary_accuracy"


class Accuracy(BinaryOrMultiLoss, ClassificationLossFunction):
    def __init__(
        self,
        output_logits: bool = False,
        label_logits: bool = False,
        hard: bool = False,
    ):
        BinaryOrMultiLoss.__init__(
            self,
            BinaryAccuracy(output_logits, label_logits, hard=hard),
            MultiAccuracy(output_logits, label_logits, hard=hard),
            name="accuracy",
        )
        ClassificationLossFunction.__init__(self, output_logits, label_logits)
        self.hard = hard
