from dataclasses import dataclass
from pathlib import Path
from tqdm import trange  # type: ignore[import-untyped]
import numpy as np
import torch
import torch.utils.data as data

from ..params import (
    EMBEDDINGS_DIR,
    LABELS_DIR,
    DatasetName,
    ModelName,
    CPU_GENERATOR,
    DEFAULT_DEVICE,
)
from ..file_utils import safe_open_dir
from ..np_utils import index_random_split
from ..models import load_model
from .registry import load_dataset, Dataset


@dataclass
class EmbeddingsDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    dataset_name: DatasetName
    model_name: ModelName


@dataclass
class ModelLabeledTrainingDataset:
    y_labels: np.ndarray
    y_logits: np.ndarray
    num_classes: int
    dataset_name: DatasetName
    model_name: ModelName


def get_embeddings_path(dataset_name: DatasetName, model_name: ModelName) -> Path:
    embeddings_path = EMBEDDINGS_DIR / f"{dataset_name.value}" / f"{model_name.value}"
    return embeddings_path


def embeddings_exist(dataset_name: DatasetName, model_name: ModelName) -> bool:
    embeddings_path = get_embeddings_path(dataset_name, model_name)
    return all(
        (embeddings_path / f"{split}.npy").exists()
        for split in ["x_train", "y_train", "x_test", "y_test"]
    )


def load_embeddings(
    dataset_name: DatasetName,
    model_name: ModelName,
    filter_list: np.ndarray | None = None,
) -> EmbeddingsDataset:
    embeddings_path = get_embeddings_path(dataset_name, model_name)
    x_train: np.ndarray = np.load(embeddings_path / "x_train.npy")
    y_train: np.ndarray = np.load(embeddings_path / "y_train.npy")
    x_test: np.ndarray = np.load(embeddings_path / "x_test.npy")
    y_test: np.ndarray = np.load(embeddings_path / "y_test.npy")
    num_classes = max(y_train.max(), y_test.max()) + 1
    dset = EmbeddingsDataset(
        x_train,
        y_train,
        x_test,
        y_test,
        num_classes=num_classes,
        dataset_name=dataset_name,
        model_name=model_name,
    )

    if filter_list is not None:
        dset = filter_labels(dset, filter_list)

    return dset


def get_labels_path(
    dataset_name: DatasetName, model_name: ModelName, seed: int | None = None
):
    labels_path = LABELS_DIR / f"{dataset_name.value}" / f"{model_name.value}"
    if seed is not None:
        labels_path = labels_path / f"{seed}"
    return labels_path


def load_model_labels(
    dataset_name: DatasetName, model_name: ModelName, seed: int | None = None
) -> ModelLabeledTrainingDataset:
    labels_path = get_labels_path(dataset_name, model_name, seed=seed)
    y_labels: np.ndarray = np.load(labels_path / "y_labels.npy")
    y_logits: np.ndarray = np.load(labels_path / "y_logits.npy")
    num_classes = y_logits.shape[1]
    return ModelLabeledTrainingDataset(
        y_labels,
        y_logits,
        num_classes=num_classes,
        dataset_name=dataset_name,
        model_name=model_name,
    )


def most_common_classes(k: int, labels: np.ndarray) -> np.ndarray:
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_labels = unique_labels[np.argsort(counts)[::-1]]
    most_common_labels = sorted_labels[:k]
    return most_common_labels


def filter_labels(
    dset: EmbeddingsDataset, filter_list: np.ndarray
) -> EmbeddingsDataset:
    new_y_train = np.full_like(dset.y_train, fill_value=-1)
    new_y_test = np.full_like(dset.y_test, fill_value=-1)
    for i, label in enumerate(filter_list):
        new_y_train[dset.y_train == label] = i
        new_y_test[dset.y_test == label] = i

    # Remove samples with -1 labels
    train_mask = new_y_train != -1
    test_mask = new_y_test != -1
    x_train = dset.x_train[train_mask]
    y_train = new_y_train[train_mask]
    x_test = dset.x_test[test_mask]
    y_test = new_y_test[test_mask]

    return EmbeddingsDataset(
        x_train,
        y_train,
        x_test,
        y_test,
        num_classes=len(filter_list),
        dataset_name=dset.dataset_name,
        model_name=dset.model_name,
    )


def compute_embeddings(
    dataset_name: DatasetName,
    model_name: ModelName,
    batch_size: int = 256,
    shuffle: bool = True,
    n_train: int = 20000,
    n_test: int = 10000,
    device: torch.device = DEFAULT_DEVICE,
) -> EmbeddingsDataset:
    model = load_model(model_name, device=device)
    if dataset_name == DatasetName.IMAGENET:
        # We need special logic
        SPLIT_SEED = 666
        dataset = load_dataset(dataset_name, split="val")
        test_inds, train_inds = index_random_split(
            len(dataset),
            split_sizes=[0.2, 0.8],
            random_state=np.random.default_rng(SPLIT_SEED),
        )
        train_dataset = data.Subset(dataset, train_inds)  # type: ignore[arg-type]
        test_dataset = data.Subset(dataset, test_inds)  # type: ignore[arg-type]
    else:
        train_dataset = load_dataset(dataset_name, split="train")
        test_dataset = load_dataset(dataset_name, split="val")
    embeddings_path = get_embeddings_path(dataset_name, model_name)

    def obtain_embeddings(
        dataset: Dataset, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        loader = data.DataLoader(
            dataset, shuffle=shuffle, batch_size=batch_size, generator=CPU_GENERATOR
        )
        if len(dataset) < n_samples:
            print(
                f"Warning: dataset has {len(dataset)} samples, but {n_samples} requested. Using all samples."
            )
            n_samples = len(dataset)
        with torch.no_grad():
            x_embeddings, y_labels = None, None
            for start, (inputs, labels) in zip(
                trange(0, n_samples, batch_size), loader
            ):
                if start + batch_size > n_samples:
                    inputs = inputs[: n_samples - start]
                    labels = labels[: n_samples - start]

                inputs = inputs.to(device)
                outputs: torch.Tensor = model(inputs)

                # Convert to numpy
                np_outputs: np.ndarray = outputs.cpu().numpy()
                np_labels: np.ndarray = labels.numpy()

                # Use the first output to infer appropriate shapes
                if x_embeddings is None:
                    assert y_labels is None
                    x_embeddings = np.empty(
                        (n_samples, *np_outputs.shape[1:]), dtype=np_outputs.dtype
                    )
                    y_labels = np.empty(
                        (n_samples, *np_labels.shape[1:]), dtype=np_labels.dtype
                    )

                end = start + np_outputs.shape[0]
                assert x_embeddings is not None
                assert y_labels is not None
                x_embeddings[start:end] = np_outputs
                y_labels[start:end] = np_labels

        assert x_embeddings is not None
        assert y_labels is not None

        return x_embeddings, y_labels

    x_train, y_train = obtain_embeddings(train_dataset, n_train)  # type: ignore[arg-type]
    x_test, y_test = obtain_embeddings(test_dataset, n_test)  # type: ignore[arg-type]

    print(f"Saving embeddings to {embeddings_path}")
    np.save(safe_open_dir(embeddings_path) / "x_train.npy", x_train)
    np.save(safe_open_dir(embeddings_path) / "y_train.npy", y_train)
    np.save(safe_open_dir(embeddings_path) / "x_test.npy", x_test)
    np.save(safe_open_dir(embeddings_path) / "y_test.npy", y_test)

    num_classes = max(y_train.max(), y_test.max()) + 1

    return EmbeddingsDataset(
        x_train,
        y_train,
        x_test,
        y_test,
        num_classes=num_classes,
        dataset_name=dataset_name,
        model_name=model_name,
    )
