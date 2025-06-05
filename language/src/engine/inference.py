from typing import Iterator, Optional, Sequence

from collections import deque
import torch.utils
from tqdm import tqdm, trange  # type: ignore[import-untyped]

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.utils.data as data

from ..params import Dataset, CPU
from ..file_utils import safe_open_file
from ..datasets import LabeledDataBatch, DataSampler
from ..metrics.losses import LossFunction
from ..measurements import MonteCarloEstimator


class Inference:

    def __init__(
        self,
        model: nn.Module,
    ):
        self.model = model

        self.device = next(model.parameters()).device
        print(f"Using model device for inference: {self.device}")

    @torch.no_grad()
    def inference_step(self, data_batch: LabeledDataBatch) -> torch.Tensor:
        """
        Assumes model is already in train mode if desired
        """

        # Convert data bach to device
        data_batch = data_batch.to(self.device)

        out = self.model(data_batch.x)
        return out.to(CPU)


class DatasetInference(Inference):
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
    ):
        super(DatasetInference, self).__init__(
            model,
        )
        self.dataset = dataset

    def inference(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        data_loader = data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,  # Do not shuffle for inference
            drop_last=False,
        )
        self.model.eval()
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        num_samples = len(self.dataset)
        outs: Optional[torch.Tensor] = None
        for i, tensors in pbar:
            data_batch = LabeledDataBatch.read_data_batch(tensors, device=self.device)
            out = self.inference_step(data_batch)
            if outs is None:
                outs = torch.empty(
                    (num_samples, *out.shape[1:]), device=CPU, dtype=out.dtype
                )

            assert outs is not None
            start = data_loader.batch_size * i
            end = start + out.shape[0]
            outs[start:end] = out

        assert outs is not None
        return outs
