import numpy as np

from src.params import ModelName, DatasetName
from src.datasets import load_embeddings


def test_all_embeddings_match():
    for dataset_name in [DatasetName.CIFAR10, DatasetName.IMAGENET]:
        print(f"Testing dataset: {dataset_name}")
        alexnet_embeddings = load_embeddings(dataset_name, ModelName.ALEXNET)
        resnet50_dino_embeddings = load_embeddings(
            dataset_name, ModelName.RESNET50_DINO
        )
        vitb8_dino_embeddings = load_embeddings(dataset_name, ModelName.VITB8_DINO)

        assert (
            alexnet_embeddings.x_train.shape[0]
            == resnet50_dino_embeddings.x_train.shape[0]
        )
        assert (
            alexnet_embeddings.x_train.shape[0]
            == vitb8_dino_embeddings.x_train.shape[0]
        )
        assert (
            alexnet_embeddings.x_test.shape[0]
            == resnet50_dino_embeddings.x_test.shape[0]
        )
        assert (
            alexnet_embeddings.x_test.shape[0] == vitb8_dino_embeddings.x_test.shape[0]
        )
        assert alexnet_embeddings.num_classes == resnet50_dino_embeddings.num_classes
        assert alexnet_embeddings.num_classes == vitb8_dino_embeddings.num_classes
        assert alexnet_embeddings.dataset_name == resnet50_dino_embeddings.dataset_name
        assert alexnet_embeddings.dataset_name == vitb8_dino_embeddings.dataset_name
        assert alexnet_embeddings.model_name == ModelName.ALEXNET
        assert resnet50_dino_embeddings.model_name == ModelName.RESNET50_DINO
        assert vitb8_dino_embeddings.model_name == ModelName.VITB8_DINO
        assert np.all(alexnet_embeddings.y_train == resnet50_dino_embeddings.y_train)
        assert np.all(alexnet_embeddings.y_train == vitb8_dino_embeddings.y_train)
        assert np.all(alexnet_embeddings.y_test == resnet50_dino_embeddings.y_test)
        assert np.all(alexnet_embeddings.y_test == vitb8_dino_embeddings.y_test)

        print(f"All embeddings for {dataset_name} match!")
