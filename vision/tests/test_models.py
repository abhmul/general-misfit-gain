import torch
import itertools as it

from src.params import DatasetName, ModelName, MODELS_DIR, CPU
from src.models import get_alexnet_classifier


def get_weak_path(dataset_name: DatasetName, weak_model_name: ModelName):
    return MODELS_DIR / f"{dataset_name.value}_{weak_model_name.value}.pt"


def get_st_gt_glob(dataset_name: DatasetName, model_name: ModelName):
    return list(MODELS_DIR.glob(f"{dataset_name.value}_{model_name.value}_*_stgt.pt"))


def get_st_path(
    dataset_name: DatasetName,
    model_name: ModelName,
):
    return list(MODELS_DIR.glob(f"{dataset_name.value}_{model_name.value}_*_st.pt"))


def test_models_agree():
    for dataset_name in [DatasetName.CIFAR10, DatasetName.IMAGENET]:
        print(f"Testing dataset: {dataset_name}")
        alexnet_state_dict = torch.load(
            get_weak_path(dataset_name, ModelName.ALEXNET), weights_only=True
        )

        resnet50_dino_stgt_state_dicts = []
        for stgt_path in get_st_gt_glob(dataset_name, ModelName.RESNET50_DINO):
            resnet50_dino_stgt_state_dicts.append(
                {
                    "model_name": ModelName.RESNET50_DINO,
                    "type": "stgt",
                    "state_dict": torch.load(
                        stgt_path, weights_only=True, map_location=CPU
                    ),
                }
            )
        vitb8_dino_stgt_state_dicts = []
        for stgt_path in get_st_gt_glob(dataset_name, ModelName.VITB8_DINO):
            vitb8_dino_stgt_state_dicts.append(
                {
                    "model_name": ModelName.VITB8_DINO,
                    "type": "stgt",
                    "state_dict": torch.load(
                        stgt_path, weights_only=True, map_location=CPU
                    ),
                }
            )

        resnet50_dino_st_state_dicts = []
        for st_path in get_st_path(dataset_name, ModelName.RESNET50_DINO):
            resnet50_dino_st_state_dicts.append(
                {
                    "model_name": ModelName.RESNET50_DINO,
                    "type": "st",
                    "state_dict": torch.load(
                        st_path, weights_only=True, map_location=CPU
                    ),
                }
            )
        vitb8_dino_st_state_dicts = []
        for st_path in get_st_path(dataset_name, ModelName.VITB8_DINO):
            vitb8_dino_st_state_dicts.append(
                {
                    "model_name": ModelName.VITB8_DINO,
                    "type": "st",
                    "state_dict": torch.load(
                        st_path, weights_only=True, map_location=CPU
                    ),
                }
            )

        # Check that all have the same weak_inds and st_inds
        for st_model in it.chain(
            resnet50_dino_stgt_state_dicts,
            vitb8_dino_stgt_state_dicts,
            resnet50_dino_st_state_dicts,
            vitb8_dino_st_state_dicts,
        ):
            print(
                f"Checking inds of {st_model['model_name']} of type {st_model['type']}"
            )
            state_dict = st_model["state_dict"]
            assert torch.all(state_dict["weak_inds"] == alexnet_state_dict["weak_inds"])
            assert torch.all(state_dict["st_inds"] == alexnet_state_dict["st_inds"])

            state_dict.pop("weak_inds")
            state_dict.pop("st_inds")
            set(state_dict.keys()) == {"finetune.weight", "finetune.bias"}

        if dataset_name == DatasetName.IMAGENET:
            expected_alexnet = get_alexnet_classifier().to(CPU).state_dict()
            alexnet_state_dict.pop("weak_inds")
            alexnet_state_dict.pop("st_inds")
            assert alexnet_state_dict.keys() == expected_alexnet.keys()
            assert all(
                torch.allclose(alexnet_state_dict[k], expected_alexnet[k])
                for k in alexnet_state_dict
            )
        else:
            alexnet_state_dict.pop("weak_inds")
            alexnet_state_dict.pop("st_inds")
            set(alexnet_state_dict.keys()) == {"finetune.weight", "finetune.bias"}

        print(f"All trained models for {dataset_name} match!")
