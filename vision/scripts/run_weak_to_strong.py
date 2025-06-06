from pathlib import Path
from typing import Literal, Optional
from argparse import ArgumentParser
from pprint import pprint
import json

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from torchinfo import summary

from src.params import (
    ModelName,
    DatasetName,
    get_new_seed,
    set_seed,
    DeviceType,
    CPU_GENERATOR,
    RESULTS_DIR,
    MODELS_DIR,
    CPU,
    DEFAULT_DEVICE,
)
from src.np_utils import index_random_split
from src.py_utils import JsonType
from src.file_utils import generate_file_tag, safe_open_file
from src.torch_utils import sigmoid_or_softmax

from src.datasets import (
    load_embeddings,
    EmbeddingsDataset,
    most_common_classes,
    filter_labels,
)
from src.models import LogisticRegression, get_alexnet_classifier, SelectOutputs
from src.metrics import CrossEntropy, KLDivergence, Accuracy, ClassificationLossFunction
from src.engine import DatasetClassificationTrainer, DatasetInference
from src.measurements import EstimateWeakToStrong, ModuleSpec

parser = ArgumentParser(
    """ONLY VISION DATASETS SUPPORTED. Please note this is seed dependent."""
)
parser.add_argument(
    "--weak_model", type=ModelName, default=ModelName.ALEXNET, choices=list(ModelName)
)
parser.add_argument(
    "--strong_model",
    type=ModelName,
    default=ModelName.RESNET50_DINO,
    choices=list(ModelName),
)
parser.add_argument(
    "--dataset",
    type=DatasetName,
    default=DatasetName.CIFAR10,
    choices=list(DatasetName),
)
parser.add_argument("--weak_split", type=float, default=0.5)
parser.add_argument("--weak_validation_split", type=float, default=0.0)
parser.add_argument("--st_validation_split", type=float, default=0.2)
parser.add_argument("--num_labels", type=int)

parser.add_argument("--use_alexnet_classifier", action="store_true")

# Weak training settings
parser.add_argument("--retrain_weak", action="store_true")
parser.add_argument("--weak_weight_decay", type=float, default=0.0)
parser.add_argument("--weak_lr", type=float, default=1e-3)
parser.add_argument("--weak_epochs", type=int, default=20)
parser.add_argument("--weak_batch_size", type=int, default=256)

# Strong training settings
parser.add_argument("--retrain_stgt", action="store_true")
parser.add_argument("--strong_weight_decay", type=float, default=0.1)
parser.add_argument("--strong_lr", type=float, default=1e-3)
parser.add_argument("--strong_epochs", type=int, default=200)
parser.add_argument("--strong_batch_size", type=int, default=256)
parser.add_argument("--num_heads", type=int, default=1)
parser.add_argument("--stgt_num_heads", type=int, default=1)
parser.add_argument("--stgt_lr", type=float, default=1e-3)
parser.add_argument("--stgt_epochs", type=int, default=200)
parser.add_argument(
    "--forward",
    action="store_true",
    help="Use forward training for st and stgt",
)
parser.add_argument("--reuse_st", action="store_true")

# Other training settings
parser.add_argument(
    "--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "adam"]
)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--exp_id", type=str)


def embeddings_to_dataset(
    embeddings: EmbeddingsDataset,
    inds: Optional[np.ndarray] = None,
) -> data.TensorDataset:
    x = embeddings.x_train[inds] if inds is not None else embeddings.x_train
    y = embeddings.y_train[inds] if inds is not None else embeddings.y_train
    return data.TensorDataset(
        torch.tensor(x),
        F.one_hot(torch.tensor(y), num_classes=embeddings.num_classes).to(
            torch.float32
        ),
    )


def with_weak_labels(
    embeddings: EmbeddingsDataset,
    weak_labels: torch.Tensor,
    weak_logits: torch.Tensor,
    inds: Optional[np.ndarray] = None,
) -> data.TensorDataset:
    x = embeddings.x_train[inds] if inds is not None else embeddings.x_train
    weak_labels = weak_labels[inds] if inds is not None else weak_labels
    weak_logits = weak_logits[inds] if inds is not None else weak_logits
    return data.TensorDataset(torch.tensor(x), weak_labels, weak_logits)


def get_weak_path(dataset_name: DatasetName, weak_model_name: ModelName):
    return MODELS_DIR / f"{dataset_name.value}_{weak_model_name.value}.pt"


def _strong_path(
    dataset_name: DatasetName,
    model_name: ModelName,
    num_heads: int,
    forward: bool | None = None,
    seed: int | None = None,
    num_labels: int | None = None,
):
    tag_terms = [dataset_name.value, model_name.value, str(num_heads)]
    if forward:
        # If using forward training, we need to specify that in the tag
        tag_terms.append("forward")
    if seed is not None:
        tag_terms.append(str(seed))
    if num_labels is not None:
        tag_terms.insert(1, str(num_labels))
    tag = "_".join(tag_terms)
    return tag


def get_st_gt_path(
    dataset_name: DatasetName,
    model_name: ModelName,
    num_heads: int,
    num_labels: int | None = None,
):
    return (
        MODELS_DIR
        / f"{_strong_path(dataset_name, model_name, num_heads, num_labels=num_labels)}_stgt.pt"
    )


def get_st_path(
    dataset_name: DatasetName,
    model_name: ModelName,
    num_heads: int,
    forward: bool,
    seed: int,
    num_labels: int | None,
):
    tag = _strong_path(
        dataset_name,
        model_name,
        num_heads,
        forward=forward,
        seed=seed,
        num_labels=num_labels,
    )
    return MODELS_DIR / f"{tag}_st.pt"


def load_alexnet_classifier(
    weak_model_name: ModelName,
    dataset_name: DatasetName,
    device: torch.device,
    classes: np.ndarray | None = None,
):
    assert weak_model_name == ModelName.ALEXNET
    assert dataset_name == DatasetName.IMAGENET
    weak_model = get_alexnet_classifier().to(device)
    if classes is not None:
        weak_model = SelectOutputs(weak_model, torch.tensor(classes, dtype=torch.long))
    return weak_model


def load_weak_model(
    dataset_name: DatasetName,
    weak_model_name: ModelName,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    use_alexnet_classifier: bool = False,
    classes: np.ndarray | None = None,
):
    weak_path = get_weak_path(dataset_name, weak_model_name)
    assert weak_path.exists()
    state_dict = torch.load(safe_open_file(weak_path), weights_only=True)

    weak_inds: torch.Tensor = state_dict.pop("weak_inds")
    st_inds: torch.Tensor = state_dict.pop("st_inds")

    if use_alexnet_classifier:
        print("Using AlexNet pretrained classifier as weak model")
        weak_model = load_alexnet_classifier(
            weak_model_name=weak_model_name,
            dataset_name=dataset_name,
            device=device,
            classes=classes,
        )
    else:
        weak_model = LogisticRegression(
            input_dim=input_dim, output_dim=output_dim, num_heads=1, logit_outputs=True
        )
        weak_model.load_state_dict(state_dict, strict=True)
        weak_model = weak_model.to(device)
    weak_model.eval()

    return weak_model, weak_inds.numpy(), st_inds.numpy()


def load_strong_model(
    dataset_name: DatasetName,
    st_model_name: ModelName,
    num_heads: int,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    forward: bool,
    seed: int,
    stgt: bool,
    num_labels: int | None = None,
):
    if stgt:
        assert forward, "Forward training must be used for stgt models."
        st_path = get_st_gt_path(
            dataset_name,
            st_model_name,
            num_heads,
            num_labels=num_labels,
        )
    else:
        st_path = get_st_path(
            dataset_name,
            st_model_name,
            num_heads,
            forward=forward,
            seed=seed,
            num_labels=num_labels,
        )
    assert st_path.exists()
    state_dict = torch.load(safe_open_file(st_path), weights_only=True)

    weak_inds: torch.Tensor = state_dict.pop("weak_inds")
    st_inds: torch.Tensor = state_dict.pop("st_inds")

    st_model = LogisticRegression(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        logit_outputs=True,
    )
    st_model.load_state_dict(state_dict, strict=True)
    st_model = st_model.to(device)

    st_model.eval()

    return st_model, weak_inds.numpy(), st_inds.numpy()


def train_model(
    input_dim: int,
    output_dim: int,
    num_heads: int,
    dataset: data.TensorDataset,
    val_dataset: data.TensorDataset | None,
    weight_decay: float,
    lr: float,
    batch_size: int,
    num_epochs: int,
    loss: ClassificationLossFunction,
    metrics: list[ClassificationLossFunction],
    optimizer_name: Literal["adamw", "sgd", "adam"],
    device: torch.device,
):
    model = LogisticRegression(
        input_dim=input_dim, output_dim=output_dim, num_heads=num_heads
    )
    model = model.to(device)
    summary(model, input_size=(batch_size, input_dim))

    # n_iter = num_epochs * len(range(0, len(dataset), batch_size))
    # Apply weight decay only to linear weights
    separate_params = model.weight_decay_params()
    if optimizer_name == "adamw":
        optimizer: optim.Optimizer = optim.AdamW(
            [
                {
                    "params": separate_params["decay_params"],
                    "weight_decay": weight_decay,
                },
                {"params": separate_params["other_params"], "weight_decay": 0.0},
            ],
            lr=lr,
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            [
                {
                    "params": separate_params["decay_params"],
                    "weight_decay": weight_decay,
                },
                {"params": separate_params["other_params"], "weight_decay": 0.0},
            ],
            lr=lr,
        )
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            [
                {
                    "params": separate_params["decay_params"],
                    "weight_decay": weight_decay,
                },
                {"params": separate_params["other_params"], "weight_decay": 0.0},
            ],
            lr=lr,
        )
    # schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)
    # optimizer = optim.SGD(model.parameters(), weight_decay=weight_decay, lr=lr)
    # schedule = None

    trainer = DatasetClassificationTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        dataset=dataset,  # type: ignore[arg-type]
        val_dataset=val_dataset,  # type: ignore[arg-type]
        metrics=metrics,
        # scheduler=schedule,
    )
    trainer.train(num_epochs=num_epochs, batch_size=batch_size)

    return model


def make_validation_split(
    inds: np.ndarray, embeddings: EmbeddingsDataset, validation_split: float
) -> tuple[data.TensorDataset, data.TensorDataset | None, np.ndarray, np.ndarray]:
    np_rng = np.random.default_rng(seed=get_new_seed())
    if validation_split == 0:
        val_inds = np.arange(0)
        train_inds = inds
        val_dataset = None
    else:
        val_inds, train_inds = index_random_split(
            inds,
            split_sizes=[validation_split, 1 - validation_split],
            random_state=np_rng,
        )
        val_dataset = embeddings_to_dataset(embeddings, inds=val_inds)

    train_dataset = embeddings_to_dataset(embeddings, inds=train_inds)

    return train_dataset, val_dataset, train_inds, val_inds


def train_weak_model(
    dataset_name: DatasetName,
    weak_model_name: ModelName,
    weak_embeddings: EmbeddingsDataset,
    weak_split: float,
    validation_split: float,
    use_alexnet_classifier: bool,
    weak_weight_decay: float,
    weak_lr: float,
    weak_batch_size: int,
    weak_epochs: int,
    optimzer_name: Literal["adamw", "sgd", "adam"],
    device=torch.device,
    classes: np.ndarray | None = None,
):
    # Split into two parts - these are fixed on subsequent runs
    np_rng = np.random.default_rng(seed=get_new_seed())
    if use_alexnet_classifier:
        # We don't need to split the data
        weak_inds = np.arange(0)
        st_inds = np_rng.permutation(weak_embeddings.x_train.shape[0])

        print("Using AlexNet pretrained classifier as weak model")
        # This is not just a simple logistic regression
        weak_model = load_alexnet_classifier(
            weak_model_name=weak_model_name,
            dataset_name=dataset_name,
            device=device,
            classes=classes,
        )
    else:
        weak_inds, st_inds = index_random_split(
            weak_embeddings.x_train.shape[0],
            split_sizes=[weak_split, 1 - weak_split],
            random_state=np_rng,
        )
        # No validation split for weak model to match NLP experiments
        weak_train_dataset, weak_val_dataset, _, _ = make_validation_split(
            inds=weak_inds,
            embeddings=weak_embeddings,
            validation_split=validation_split,
        )
        loss_fn = CrossEntropy(output_logits=True, label_logits=False)
        accuracy_fn = Accuracy(output_logits=True, label_logits=False)
        print("Training weak model from scratch")
        weak_model = train_model(
            input_dim=weak_embeddings.x_train.shape[1],
            output_dim=weak_embeddings.num_classes,
            num_heads=1,
            dataset=weak_train_dataset,
            val_dataset=weak_val_dataset,
            weight_decay=weak_weight_decay,
            lr=weak_lr,
            batch_size=weak_batch_size,
            num_epochs=weak_epochs,
            loss=loss_fn,
            metrics=[accuracy_fn],
            optimizer_name=optimzer_name,
            device=device,
        )

    return weak_model, weak_inds, st_inds


def save_model(
    model: torch.nn.Module, path: Path, weak_inds: np.ndarray, st_inds: np.ndarray
):
    state_dict = model.state_dict()
    state_dict["weak_inds"] = torch.tensor(weak_inds)
    state_dict["st_inds"] = torch.tensor(st_inds)
    torch.save(state_dict, safe_open_file(path))


if __name__ == "__main__":
    args = parser.parse_args()
    pprint(args)

    # This will allow us to get new runs that are still reproducible
    set_seed(args.seed)

    # For CIFAR10 no real speedup from GPU unless using multiple heads
    DEVICE = CPU if args.use_cpu else DEFAULT_DEVICE

    # file tag
    file_tag = generate_file_tag(
        Path(__file__).stem,
        "wk",
        args.weak_model.value,
        "st",
        args.strong_model.value,
        "nh",
        args.num_heads,
        "stgt_nh",
        args.stgt_num_heads,
        "ds",
        args.dataset.value,
        *(["nl", str(args.num_labels)] if args.num_labels is not None else []),
        # Add forward if using forward training
        *(["forward"] if args.forward else []),
        "seed",
        str(args.seed),
    )
    # Settings
    settings: dict[str, JsonType] = {k: str(v) for k, v in vars(args).items()}
    # Add additional tags used for distinguishing versions of this script
    settings["version"] = 6.0
    settings["debug"] = args.debug
    settings["exp_id"] = args.exp_id
    settings["weight_decay_fixed"] = True

    ### WEAK LOADING TRAINING ###
    weak_embeddings = load_embeddings(
        dataset_name=args.dataset, model_name=args.weak_model, filter_list=None
    )

    # Filter the labels if needed
    # -- this can be used to make the imagenet problem easier for debugging
    # and exploring dependence of the misfit inequality on the number of classes.
    # However, it will reduce the number of samples, so this may not be ideal if there aren't
    # that many samples already.
    most_common = None
    if args.num_labels is not None:
        all_labels = np.concatenate([weak_embeddings.y_train, weak_embeddings.y_test])
        most_common = most_common_classes(args.num_labels, all_labels)
        weak_embeddings = filter_labels(weak_embeddings, most_common)

    # Load weak model and train if it doesn't exist
    weak_path = get_weak_path(
        dataset_name=args.dataset, weak_model_name=args.weak_model
    )
    if not args.retrain_weak and weak_path.exists():
        print(f"Found weak model at {weak_path}, loading...")
        weak_model, weak_inds, st_inds = load_weak_model(
            dataset_name=args.dataset,
            weak_model_name=args.weak_model,
            input_dim=weak_embeddings.x_train.shape[1],
            output_dim=weak_embeddings.num_classes,
            device=DEVICE,
            use_alexnet_classifier=args.use_alexnet_classifier,
            classes=most_common,
        )
    else:
        if args.retrain_weak and weak_path.exists():
            print("Retraining weak model...")
        else:
            print(f"Did not find weak model at {weak_path}, training...")

        weak_model, weak_inds, st_inds = train_weak_model(
            dataset_name=args.dataset,
            weak_model_name=args.weak_model,
            weak_embeddings=weak_embeddings,
            weak_split=args.weak_split,
            validation_split=args.weak_validation_split,
            use_alexnet_classifier=args.use_alexnet_classifier,
            weak_weight_decay=args.weak_weight_decay,
            weak_lr=args.weak_lr,
            weak_batch_size=args.weak_batch_size,
            weak_epochs=args.weak_epochs,
            device=DEVICE,
            optimzer_name=args.optimizer,
            classes=most_common,
        )

        # Save the weak model
        print(f"Saving weak model to {weak_path}")
        save_model(weak_model, weak_path, weak_inds, st_inds)

    print("Weak model summary:")
    summary(
        weak_model,
        input_size=(args.weak_batch_size, weak_embeddings.x_train.shape[1]),
    )
    # Compute weak labels
    weak_full_dataset = embeddings_to_dataset(weak_embeddings)
    weak_inference = DatasetInference(
        weak_model,
        weak_full_dataset,  # type: ignore[arg-type]
    )
    weak_logits = weak_inference.inference(batch_size=args.weak_batch_size)
    weak_labels = sigmoid_or_softmax(weak_logits, dim=1)

    # Free some memory
    del weak_embeddings
    del weak_full_dataset
    del weak_inference
    del weak_model
    ### END WEAK LOADING TRAINING ###

    ### ST_GT LOADING AND TRAINING ###
    # Do the gt->strong training with selection of labels
    strong_embeddings_for_gt = load_embeddings(
        model_name=args.strong_model, dataset_name=args.dataset, filter_list=most_common
    )

    # Save the st gt model
    st_gt_path = get_st_gt_path(
        dataset_name=args.dataset,
        model_name=args.strong_model,
        num_heads=args.stgt_num_heads,
        num_labels=args.num_labels,
    )

    if not args.retrain_stgt and st_gt_path.exists():
        print(f"Found stgt model at {st_gt_path}, loading...")
        strong_gt_model, weak_inds, st_inds = load_strong_model(
            dataset_name=args.dataset,
            st_model_name=args.strong_model,
            num_heads=args.stgt_num_heads,
            input_dim=strong_embeddings_for_gt.x_train.shape[1],
            output_dim=strong_embeddings_for_gt.num_classes,
            device=DEVICE,
            forward=True,
            seed=args.seed,
            stgt=True,
            num_labels=args.num_labels,
        )
        summary(
            strong_gt_model,
            input_size=(
                args.strong_batch_size,
                strong_embeddings_for_gt.x_train.shape[1],
            ),
        )
    else:
        if args.retrain_stgt and st_gt_path.exists():
            print("Retraining stgt model...")
        else:
            print(f"Did not find stgt model at {st_gt_path}, training...")

        # Train the strong gt model on the same data as the weak model
        # If weak model was alexnet and no training required, just use all the data.
        st_gt_train_dataset, st_gt_val_dataset, st_gt_train_inds, st_gt_val_inds = (
            make_validation_split(
                weak_inds if len(weak_inds) > 0 else st_inds,
                strong_embeddings_for_gt,
                args.st_validation_split,
            )
        )
        print(f"Training strong model {args.strong_model} on ground truth...")
        strong_gt_model = train_model(
            input_dim=strong_embeddings_for_gt.x_train.shape[1],
            output_dim=strong_embeddings_for_gt.num_classes,
            num_heads=args.stgt_num_heads,
            dataset=st_gt_train_dataset,
            val_dataset=st_gt_val_dataset,
            weight_decay=args.strong_weight_decay,
            lr=args.stgt_lr,
            batch_size=args.strong_batch_size,
            num_epochs=args.stgt_epochs,
            loss=CrossEntropy(output_logits=True, label_logits=False),
            metrics=[Accuracy(output_logits=False, label_logits=False)],
            optimizer_name=args.optimizer,
            device=DEVICE,
        )

        print(f"Saving strong-GT model to {st_gt_path}")
        save_model(strong_gt_model, st_gt_path, weak_inds, st_inds)

        # Free some memory
        del st_gt_train_dataset
        del st_gt_val_dataset
        del st_gt_train_inds
        del st_gt_val_inds

    del strong_gt_model
    del strong_embeddings_for_gt
    ### END ST_GT LOADING AND TRAINING ###

    ## WEAK->STRONG LOADING AND TRAINING ##
    # Do the weak->strong training
    strong_embeddings = load_embeddings(
        model_name=args.strong_model, dataset_name=args.dataset, filter_list=most_common
    )

    if args.st_validation_split == 0:
        weakly_labeled_val_dataset = None
        st_train_inds = st_inds
    else:
        _, __, st_train_inds, st_val_inds = make_validation_split(
            inds=st_inds,
            embeddings=strong_embeddings,
            validation_split=args.st_validation_split,
        )
        weakly_labeled_val_dataset = with_weak_labels(
            strong_embeddings, weak_labels, weak_logits, inds=st_val_inds
        )
    weakly_labeled_train_dataset = with_weak_labels(
        strong_embeddings, weak_labels, weak_logits, inds=st_train_inds
    )
    if not args.reuse_st:
        print(
            f"Training strong model {args.strong_model} from weak model {args.weak_model}..."
        )

        if args.forward:
            strong_loss: ClassificationLossFunction = CrossEntropy(
                output_logits=True, label_logits=True
            )
        else:
            strong_loss = KLDivergence(
                output_logits=True,
                label_logits=True,
            )

        strong_from_wk_model = train_model(
            input_dim=strong_embeddings.x_train.shape[1],
            output_dim=strong_embeddings.num_classes,
            num_heads=args.num_heads,
            dataset=weakly_labeled_train_dataset,
            val_dataset=weakly_labeled_val_dataset,
            weight_decay=args.strong_weight_decay,
            lr=args.strong_lr,
            batch_size=args.strong_batch_size,
            num_epochs=args.strong_epochs,
            loss=strong_loss,
            metrics=[Accuracy(output_logits=True, label_logits=True)],
            optimizer_name=args.optimizer,
            device=DEVICE,
        )

        # Save the st model
        st_path = get_st_path(
            dataset_name=args.dataset,
            model_name=args.strong_model,
            num_heads=args.num_heads,
            forward=args.forward,
            seed=args.seed,
            num_labels=args.num_labels,
        )
        print(f"Saving strong model to {st_path}")
        save_model(strong_from_wk_model, st_path, weak_inds, st_inds)
    else:
        print("Reusing strong model from weak model training.")
        strong_from_wk_model, _, _ = load_strong_model(
            dataset_name=args.dataset,
            st_model_name=args.strong_model,
            num_heads=args.num_heads,
            input_dim=strong_embeddings.x_train.shape[1],
            output_dim=strong_embeddings.num_classes,
            device=DEVICE,
            forward=args.forward,
            seed=args.seed,
            stgt=False,
            num_labels=args.num_labels,
        )

    # Free some memory
    del weakly_labeled_train_dataset
    del weakly_labeled_val_dataset
    del weak_labels
    del weak_logits

    ### END WEAK->STRONG LOADING AND TRAINING ###
    # Estimate the misfit
    weak_embeddings = load_embeddings(
        model_name=args.weak_model, dataset_name=args.dataset, filter_list=most_common
    )
    estimate_misfit_dataset = data.TensorDataset(
        torch.tensor(strong_embeddings.x_test),
        torch.tensor(weak_embeddings.x_test),
        F.one_hot(
            torch.tensor(weak_embeddings.y_test),
            num_classes=weak_embeddings.num_classes,
        ).to(torch.float32),
    )

    # Reload models that were deleted
    print(f"Reloading stgt model at {st_gt_path}")
    strong_gt_model, _, _ = load_strong_model(
        dataset_name=args.dataset,
        st_model_name=args.strong_model,
        num_heads=args.stgt_num_heads,
        input_dim=strong_embeddings.x_test.shape[1],
        output_dim=strong_embeddings.num_classes,
        device=DEVICE,
        forward=True,  # Must be true for stgt
        seed=args.seed,
        stgt=True,
        num_labels=args.num_labels,
    )
    print("Strong-GT model summary:")
    summary(
        strong_gt_model,
        input_size=(
            args.strong_batch_size,
            strong_embeddings.x_test.shape[1],
        ),
    )

    print(f"Reloading weak model at {weak_path}, loading...")
    weak_model, _, _ = load_weak_model(
        dataset_name=args.dataset,
        weak_model_name=args.weak_model,
        input_dim=weak_embeddings.x_test.shape[1],
        output_dim=weak_embeddings.num_classes,
        device=DEVICE,
        use_alexnet_classifier=args.use_alexnet_classifier,
        classes=most_common,
    )
    print("Weak model summary:")
    summary(
        weak_model,
        input_size=(args.weak_batch_size, weak_embeddings.x_train.shape[1]),
    )

    stgt_spec = ModuleSpec(name="stgt", model=strong_gt_model)
    st_spec = ModuleSpec(name="st", model=strong_from_wk_model)
    wk_spec = ModuleSpec(name="wk", model=weak_model)
    strong_class_models = {
        stgt_spec.name: stgt_spec.model,
        st_spec.name: st_spec.model,
    }
    weak_class_models = {
        wk_spec.name: wk_spec.model,
    }
    misfit_estimator = EstimateWeakToStrong(
        dataset=estimate_misfit_dataset,  # type: ignore[arg-type]
        strong_models=strong_class_models,
        weak_models=weak_class_models,
        losses=EstimateWeakToStrong.generate_default_loss_specs(
            dual_stgt=stgt_spec, dual_st=st_spec, dual_wk=wk_spec
        ),
    )
    result = misfit_estimator.estimate(batch_size=args.strong_batch_size)

    results_fname = RESULTS_DIR / f"{file_tag}.json"
    print(f"Saving results to {results_fname}")
    output: dict[str, JsonType] = {
        "settings": settings,
        "results": result,  # type: ignore[dict-item]
    }
    with open(safe_open_file(results_fname), "w") as f:
        json.dump(output, f)
