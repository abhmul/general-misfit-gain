from argparse import ArgumentParser
from pprint import pprint

from src.params import ModelName, DatasetName
from src.datasets import compute_embeddings, embeddings_exist

parser = ArgumentParser(
    """ONLY VISION DATASETS SUPPORTED. Please note this is seed dependent. 
    Changing the seed in src/params will change the embeddings. If in doubt, 
    check that the labels are the same across the different embeddings"""
)
parser.add_argument(
    "--model", type=ModelName, default=ModelName.ALEXNET, choices=list(ModelName)
)
parser.add_argument(
    "--dataset",
    type=DatasetName,
    default=DatasetName.CIFAR10,
    choices=list(DatasetName),
)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_train", type=int, default=1000000)
parser.add_argument("--n_test", type=int, default=1000000)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument(
    "-y", "--yes", action="store_true", help="Skip any confirmation prompts"
)

if __name__ == "__main__":
    args = parser.parse_args()
    pprint(args)

    if not args.yes:
        if embeddings_exist(args.dataset, args.model):
            print("Embeddings already exist. Overwrite?")
            response = input("y/N: ").lower()
            if response != "y":
                exit()

            print("Overwriting embeddings...")

    compute_embeddings(
        args.dataset,
        args.model,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_test=args.n_test,
        shuffle=args.shuffle,
    )
