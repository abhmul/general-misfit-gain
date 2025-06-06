# python ./dump_embeddings.py --model alexnet --dataset cifar10 --batch_size 256 -y
# python ./dump_embeddings.py --model resnet50_dino --dataset cifar10 --batch_size 256 -y
# python ./dump_embeddings.py --model vitb8_dino --dataset cifar10 --batch_size 32 -y

python ./dump_embeddings.py --model alexnet --dataset imagenet --batch_size 256 -y
python ./dump_embeddings.py --model resnet50_dino --dataset imagenet --batch_size 256 -y
python ./dump_embeddings.py --model vitb8_dino --dataset imagenet --batch_size 32 -y