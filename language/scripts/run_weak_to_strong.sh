# Debug
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --retrain_weak --strong_model resnet50_dino --seed 6281 --use_cpu --debug --strong_epochs 1 --weak_epochs 1
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --retrain_weak --strong_model resnet50_dino --seed 6281 --use_cpu --debug --strong_epochs 1 --weak_epochs 1 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --retrain_weak --strong_model vitb8_dino --seed 6281 --use_cpu --debug --strong_epochs 1 --weak_epochs 1
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --retrain_weak --strong_model vitb8_dino --seed 6281 --use_cpu --debug --strong_epochs 1 --weak_epochs 1 --num_heads 100 --stgt_num_heads 100

# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier  --retrain_weak --strong_model resnet50_dino --seed 6281 --use_cpu --debug --strong_epochs 1 --weak_epochs 1
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_weak --retrain_stgt --strong_model resnet50_dino --seed 6281 --use_cpu --debug --strong_epochs 1 --strong_batch_size 64 --weak_epochs 1 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_weak --strong_model vitb8_dino --seed 6281 --use_cpu --debug --strong_epochs 1 --weak_epochs 1
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_weak --retrain_stgt --strong_model vitb8_dino --seed 6281 --use_cpu --debug --strong_epochs 1 --weak_epochs 1 --num_heads 100 --stgt_num_heads 100

# Vitb8 Cifar10 - standard w-to-s experiments
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --retrain_weak --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 6496 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 268 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 6283 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 814 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 4582 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 9371 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 6266 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 6735 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 7852 --num_heads 100 --stgt_num_heads 100

# Resnet 50 Cifar10 - standard w-to-s experiments
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6496 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 268 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6283 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 814 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 4582 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 9371 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6266 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6735 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 7852 --num_heads 100 --stgt_num_heads 100

# Vitb8 Cifar10 - vary num heads experiments
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10054 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 1963  --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 4921  --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 9977  --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 5513  --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 126   --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 6334  --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 7698  --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 7540  --num_heads 1 --stgt_num_heads 1

# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10054 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 1963 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 4921 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 9977 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 5513 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 126 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 6334 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 7698 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 7540 --num_heads 10 --stgt_num_heads 10

# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10054 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 1963 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 4921 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 9977 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 5513 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 126 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 6334 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 7698 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 7540 --num_heads 50 --stgt_num_heads 50

# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10054 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 1963 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 4921 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 9977 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 5513 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 126 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 6334 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 7698 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 7540 --num_heads 100 --stgt_num_heads 100


# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6281 --strong_epochs 200 --strong_lr 1e-3 --num_heads 100 --stgt_num_heads 1 --debug

# # Vitb8 Imagenet - vary num heads experiments
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 10054 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 1963 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 4921 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 9977 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 5513 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 126 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6334 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 7698 --num_heads 1 --stgt_num_heads 1
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 7540 --num_heads 1 --stgt_num_heads 1

# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 10054 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 1963 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 4921 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 9977 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 5513 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 126 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6334 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 7698 --num_heads 10 --stgt_num_heads 10
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 7540 --num_heads 10 --stgt_num_heads 10

# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 10054 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 1963 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 4921 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 9977 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 5513 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 126 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6334 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 7698 --num_heads 50 --stgt_num_heads 50
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 7540 --num_heads 50 --stgt_num_heads 50

# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 10054 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 1963 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 4921 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 9977 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 5513 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 126 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6334 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 7698 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --exp_id vary-num-heads --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 7540 --num_heads 100 --stgt_num_heads 100

# Vitb8 Imagenet - standard w-to-s experiments
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6496 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 268 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6283 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 814 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 4582 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 9371 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6266 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6735 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 7852 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35

# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 20 --strong_epochs 100 --strong_batch_size 64 --mode train_weak --retrain_weak --debug
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 20 --strong_epochs 100 --strong_batch_size 256 --mode train_st --debug
# Resnet 50 Imagenet - standard w-to-s experiments
# Setup weak model
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 20 --strong_epochs 100 --strong_batch_size 64 --mode train_weak --retrain_weak
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 20 --strong_epochs 100 --strong_batch_size 256 --mode train_st --debug

# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 20 --strong_epochs 200 --strong_batch_size 256
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 6496 --num_heads 100 --stgt_num_heads 100  --strong_epochs 158  --strong_batch_size 256
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 268  --num_heads 100 --stgt_num_heads 100  --strong_epochs 158  --strong_batch_size 256
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 6283 --num_heads 100 --stgt_num_heads 100  --strong_epochs 158  --strong_batch_size 256
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 814  --num_heads 100 --stgt_num_heads 100  --strong_epochs 158  --strong_batch_size 256
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 4582 --num_heads 100 --stgt_num_heads 100  --strong_epochs 158  --strong_batch_size 256
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 9371 --num_heads 100 --stgt_num_heads 100  --strong_epochs 158  --strong_batch_size 256
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 6266 --num_heads 100 --stgt_num_heads 100  --strong_epochs 158  --strong_batch_size 256
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 6735 --num_heads 100 --stgt_num_heads 100  --strong_epochs 158  --strong_batch_size 256
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model resnet50_dino --seed 7852 --num_heads 100 --stgt_num_heads 100  --strong_epochs 158  --strong_batch_size 256

# Vitb8 Imagenet - standard w-to-s experiments
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_weak --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6496 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 268 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6283 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 814 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 4582 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 9371 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6266 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6735 --num_heads 100 --stgt_num_heads 100
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 7852 --num_heads 100 --stgt_num_heads 100

# Debugging to compare to old results
## Full old settings
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 1 --stgt_num_heads 1 --stgt_epochs 20 --strong_epochs 200 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adam --strong_weight_decay 0.0
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model resnet50_dino --seed 6281 --num_heads 1 --stgt_num_heads 1 --stgt_epochs 20 --strong_epochs 200 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adam --strong_weight_decay 0.0

# With Adam, weight_decay 0.0, nh = 100
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 20 --strong_epochs 200 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adam --strong_weight_decay 0.0
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 20 --strong_epochs 200 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adam --strong_weight_decay 0.0

# With Adam, weight_decay 0.1, nh=100
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adam --strong_weight_decay 0.001
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adam --strong_weight_decay 0.001

# With fixed weight_decay 0.1, nh=100, Adamw
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1

# With fixed weight_decay 0.01 and 1.0, nh=100, Adamw
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 1.0
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 1.0
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.01
# python run_weak_to_strong.py --exp_id w-to-s-old --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.01

# Test the following with fixed weight decay
# - [ ] L2=0.1, AdamW, weak = 0.3/0.2; range k experiments for Cifar10
# - [ ] L2=0.1, AdamW, weak = 0.3/0.2; range k experiments for Imagenet
# - [ ] L2=0.1, AdamW, nh=100, weak = 0.3/0.2; w-to-s for Imagenet

# Some measurements
# # stgt - best val ep = 10
# # st - best val ep = 10
# python run_weak_to_strong.py --exp_id vary-num-heads-test --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 1 --stgt_num_heads 1 --strong_weight_decay 0.1 --weak_split 0.3 --weak_validation_split 0.2 --optimizer adamw --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 
# # stgt - best val ep = 26
# # st - best val ep = 27
# python run_weak_to_strong.py --exp_id vary-num-heads-test --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 10 --stgt_num_heads 10 --strong_weight_decay 0.1 --weak_split 0.3 --weak_validation_split 0.2 --optimizer adamw --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 
# # stgt - best val ep = 14
# # st - best val ep = 18
# python run_weak_to_strong.py --exp_id vary-num-heads-test --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 50 --stgt_num_heads 1 --strong_weight_decay 0.1 --weak_split 0.3 --weak_validation_split 0.2 --optimizer adamw --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 
# # stgt - best val ep = 14
# # st - best val ep = 18
# python run_weak_to_strong.py --exp_id vary-num-heads-test --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 100 --stgt_num_heads 100 --strong_weight_decay 0.1 --weak_split 0.3 --weak_validation_split 0.2 --optimizer adamw --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 
# # stgt - best val ep = 14
# # st - best val ep = 22
# python run_weak_to_strong.py --exp_id vary-num-heads-test --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 1000 --stgt_num_heads 1000 --strong_weight_decay 0.1 --weak_split 0.3 --weak_validation_split 0.2 --optimizer adamw --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 

# # stgt - best val ep = 6
# # st - best val ep = 13
# python run_weak_to_strong.py --exp_id vary-num-heads-test --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 1 --stgt_num_heads 1 --strong_weight_decay 0.1 --optimizer adamw --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 
# # stgt - best val ep = 6
# # st - best val ep = 17
# python run_weak_to_strong.py --exp_id vary-num-heads-test --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 10 --stgt_num_heads 10 --strong_weight_decay 0.1 --optimizer adamw --stgt_epochs 200 --strong_epochs 500 --strong_batch_size 256 
# stgt - best val ep = 6
# st - best val ep = 24
# python run_weak_to_strong.py --exp_id vary-num-heads-test --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 50 --stgt_num_heads 50 --strong_weight_decay 0.1 --optimizer adamw --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 
# stgt - best val ep = 7
# st - best val ep = 22
# python run_weak_to_strong.py --exp_id vary-num-heads-test --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 100 --stgt_num_heads 100 --strong_weight_decay 0.1 --optimizer adamw --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 
# Unable to run the following due to memory constraints
# stgt - best val ep = 
# st - best val ep = 
# python run_weak_to_strong.py --exp_id vary-num-heads-test --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 10902 --num_heads 500 --stgt_num_heads 500 --strong_weight_decay 0.1 --optimizer adamw --stgt_epochs 20 --strong_epochs 100 --strong_batch_size 1

# stgt - best val ep = 7
# st - best val ep = 24
# python run_weak_to_strong.py --exp_id w-to-s-test --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --strong_weight_decay 0.1 --optimizer adamw --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 
# stgt - best val ep = 29
# st - best val ep = 122
# python run_weak_to_strong.py --exp_id w-to-s-test --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --strong_weight_decay 0.1 --optimizer adamw --stgt_epochs 40 --strong_epochs 200 --strong_batch_size 256 

# Now rerun all vision experiments
# Resnet50 Cifar10 -- no need to retrain weak
# stgt - best val ep = 168
# st - best val ep = 54
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --retrain_stgt
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6496 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 268  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6283 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 814  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 4582 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 9371 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6266 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6735 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 7852 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1

# # Vitb8 Cifar10 -- no need to retrain weak
# # # stgt - best val ep = 14
# # # st - best val ep = 18
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --retrain_stgt
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6496 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 268  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6283 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 814  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 4582 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 9371 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6266 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6735 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 7852 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1

# # Vitb8 Cifar10 Vary k -- no need to retrain weak
# # stgt - best val ep = 10
# # st - best val ep = 10
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 10902 --num_heads 1   --stgt_num_heads 1   --stgt_epochs 20 --strong_epochs 20 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --retrain_stgt
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 10054 --num_heads 1   --stgt_num_heads 1   --stgt_epochs 20 --strong_epochs 20 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 1963  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 20 --strong_epochs 20 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 4921  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 20 --strong_epochs 20 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 9977  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 20 --strong_epochs 20 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 5513  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 20 --strong_epochs 20 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 126   --num_heads 1   --stgt_num_heads 1   --stgt_epochs 20 --strong_epochs 20 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6334  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 20 --strong_epochs 20 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 7698  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 20 --strong_epochs 20 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 7540  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 20 --strong_epochs 20 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1

# # stgt - best val ep = 26
# # st - best val ep = 27
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 10902 --num_heads 10  --stgt_num_heads 10  --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --retrain_stgt
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 10054 --num_heads 10  --stgt_num_heads 10  --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 1963  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 4921  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 9977  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 5513  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 126   --num_heads 10  --stgt_num_heads 10  --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6334  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 7698  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 7540  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 50 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1

# # # stgt - best val ep = 14
# # # st - best val ep = 18
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 10902 --num_heads 50  --stgt_num_heads 50  --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --retrain_stgt
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 10054 --num_heads 50  --stgt_num_heads 50  --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 1963  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 4921  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 9977  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 5513  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 126   --num_heads 50  --stgt_num_heads 50  --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6334  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 7698  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id vary-k-new --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 7540  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1

# Skip this and use the w-to-s run
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino    --seed 10902 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --retrain_stgt
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino    --seed 10054 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino    --seed 1963  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino    --seed 4921  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino    --seed 9977  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino    --seed 5513  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino    --seed 126   --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino    --seed 6334  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino    --seed 7698  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset cifar10 --weak_model alexnet --retrain_stgt --strong_model vitb8_dino    --seed 7540  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1

# Vitb8 Imagenet vary-k
# stgt - best val ep = 6
# st - best val ep = 13
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 10902 --num_heads 1   --stgt_num_heads 1   --stgt_epochs 15 --strong_epochs 30 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --retrain_stgt  
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 10054 --num_heads 1   --stgt_num_heads 1   --stgt_epochs 15 --strong_epochs 30 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 1963  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 15 --strong_epochs 30 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 4921  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 15 --strong_epochs 30 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 9977  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 15 --strong_epochs 30 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 5513  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 15 --strong_epochs 30 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 126   --num_heads 1   --stgt_num_heads 1   --stgt_epochs 15 --strong_epochs 30 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6334  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 15 --strong_epochs 30 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 7698  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 15 --strong_epochs 30 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 7540  --num_heads 1   --stgt_num_heads 1   --stgt_epochs 15 --strong_epochs 30 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier

# # # stgt - best val ep = 6
# # # st - best val ep = 17
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 10902 --num_heads 10  --stgt_num_heads 10  --stgt_epochs 15 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --retrain_stgt
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 10054 --num_heads 10  --stgt_num_heads 10  --stgt_epochs 15 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 1963  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 15 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 4921  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 15 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 9977  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 15 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 5513  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 15 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 126   --num_heads 10  --stgt_num_heads 10  --stgt_epochs 15 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6334  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 15 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 7698  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 15 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 7540  --num_heads 10  --stgt_num_heads 10  --stgt_epochs 15 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier

# # stgt - best val ep = 6
# # st - best val ep = 24
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 10902 --num_heads 50  --stgt_num_heads 50  --stgt_epochs 15 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --retrain_stgt
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 10054 --num_heads 50  --stgt_num_heads 50  --stgt_epochs 15 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 1963  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 15 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 4921  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 15 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 9977  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 15 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 5513  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 15 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 126   --num_heads 50  --stgt_num_heads 50  --stgt_epochs 15 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6334  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 15 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 7698  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 15 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id vary-k-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 7540  --num_heads 50  --stgt_num_heads 50  --stgt_epochs 15 --strong_epochs 50 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier

# # Vitb8 Imagenet w-to-s
# # stgt - best val ep = 7
# # st - best val ep = 24
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --retrain_stgt
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6496 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 268  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6283 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 814  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 4582 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 9371 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6266 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6735 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 7852 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier

# Resnet50 Imagenet w-to-s
# stgt - best val ep = 29
# st - best val ep = 122
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --retrain_stgt
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 6496 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 268  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 6283 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 814  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 4582 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 9371 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 6266 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 6735 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier
# python run_weak_to_strong.py --exp_id w-to-s-new --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 7852 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier

# # # Resnet50 Cifar10
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --retrain_weak --strong_model resnet50_dino --seed 6281 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6496 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 268 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6283 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 814 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 4582 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 9371 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6266 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6735 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 7852 --use_cpu

# # # VitB8 Cifar10
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --num_heads 100 --seed 1335 --strong_epochs 200 --strong_lr 2e-4 --debug
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --num_heads 100 --seed 2873 --strong_epochs 200 --strong_lr 2e-4 --debug
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 2190 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 10759 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 6500 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 10729 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 4853 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 10851 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 3577 --use_cpu
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --seed 3417 --use_cpu

# VitB8 Cifar10 test num heads
# Important here that seed is the same for all runs
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 1 --seed 1335 --strong_epochs 200 --weak_epochs 50 --retrain_weak --retrain_stgt --strong_lr 2e-4 --weak_lr 2e-4
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 2 --seed 1335 --strong_epochs 200 --strong_lr 2e-4
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 3 --seed 1335 --strong_epochs 200 --strong_lr 2e-4
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 5 --seed 1335 --strong_epochs 200 --strong_lr 2e-4
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 10 --seed 1335 --strong_epochs 200 --strong_lr 2e-4
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 20 --seed 1335 --strong_epochs 200 --strong_lr 2e-4
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 50 --seed 1335 --strong_epochs 200 --strong_lr 2e-4
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 75 --seed 1335 --strong_epochs 200 --strong_lr 2e-4
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 100 --seed 1335 --strong_epochs 200 --strong_lr 2e-4
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 200 --seed 1335 --strong_epochs 200 --strong_lr 2e-4
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino --stgt_num_heads 1 --num_heads 500 --seed 1335 --strong_epochs 200 --strong_lr 2e-4

# # Resnet 50 Imagenet
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --retrain_weak --use_alexnet_classifier --strong_epochs 400 --strong_model resnet50_dino --strong_lr 1e-4 --seed 7530
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model resnet50_dino  --strong_lr 1e-4 --seed 7133
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model resnet50_dino  --strong_lr 1e-4 --seed 8675
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model resnet50_dino  --strong_lr 1e-4 --seed 1341
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model resnet50_dino  --strong_lr 1e-4 --seed 10337
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model resnet50_dino  --strong_lr 1e-4 --seed 3491
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model resnet50_dino  --strong_lr 1e-4 --seed 2021
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model resnet50_dino  --strong_lr 1e-4 --seed 6070
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model resnet50_dino  --strong_lr 1e-4 --seed 5372
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model resnet50_dino  --strong_lr 1e-4 --seed 4524

# # VitB8 Imagenet
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 25 --strong_model vitb8_dino --num_heads 100 --strong_lr 5e-4 --seed 1199 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 25 --strong_model vitb8_dino --num_heads 100 --strong_lr 5e-4 --seed 4589 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model vitb8_dino  --strong_lr 1e-4 --seed 9676
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model vitb8_dino  --strong_lr 1e-4 --seed 5615
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model vitb8_dino  --strong_lr 1e-4 --seed 5835
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model vitb8_dino  --strong_lr 1e-4 --seed 3714
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model vitb8_dino  --strong_lr 1e-4 --seed 5318
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model vitb8_dino  --strong_lr 1e-4 --seed 6880
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model vitb8_dino  --strong_lr 1e-4 --seed 4260
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 400 --strong_model vitb8_dino  --strong_lr 1e-4 --seed 8733

# # VitB8 Imagenet - Test out different things
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --retrain_weak --use_alexnet_classifier --strong_epochs 1600 --strong_model vitb8_dino  --strong_batch_size 2048  --strong_lr 1e-5 --seed 209
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 800 --strong_model vitb8_dino  --strong_batch_size 2048  --strong_lr 1e-4 --seed 1241
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 3200 --strong_model vitb8_dino  --strong_batch_size 2048  --strong_lr 1e-6 --seed 0998
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 1600 --strong_model vitb8_dino  --strong_batch_size 4096  --strong_lr 1e-4 --seed 92746927

# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 800 --strong_model vitb8_dino  --strong_batch_size 2048  --strong_lr 1e-4 --seed 09873509 --debug --validation_split 0
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 800 --strong_model vitb8_dino  --strong_batch_size 2048  --strong_lr 1e-4 --seed 29709274294 --debug --validation_split 0.1

# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 800 --strong_model resnet50_dino  --strong_batch_size 2048  --strong_lr 1e-4 --seed 29709274294 --debug


# # VitB8 Imagenet
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 25 --strong_model vitb8_dino --num_heads 100 --num_labels 10 --strong_lr 5e-4 --seed 1199 --retrain_stgt --retrain_weak --stgt_num_heads 1 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 25 --strong_model vitb8_dino --num_heads 100 --num_labels 20 --strong_lr 5e-4 --seed 1199 --stgt_num_heads 1 --debug

# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 2000 --strong_model vitb8_dino --num_heads 100 --num_labels 10 --strong_lr 1e-3 --seed 1199 --retrain_stgt --retrain_weak --stgt_num_heads 1 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 5000 --strong_model vitb8_dino --num_heads 100 --num_labels 20 --strong_lr 1e-3 --seed 1199 --retrain_stgt --stgt_num_heads 1 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 5000 --strong_model vitb8_dino --num_heads 100 --num_labels 50 --strong_lr 1e-3 --seed 1199 --retrain_stgt --stgt_num_heads 1 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 5000 --strong_model vitb8_dino --num_heads 100 --num_labels 75 --strong_lr 1e-3 --seed 1199 --retrain_stgt --stgt_num_heads 1 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 5000 --strong_model vitb8_dino --num_heads 100 --num_labels 100 --strong_lr 1e-3 --seed 1199 --retrain_stgt --stgt_num_heads 1 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 5000 --strong_model vitb8_dino --num_heads 100 --num_labels 200 --strong_lr 1e-3 --seed 1199 --retrain_stgt --stgt_num_heads 1 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 5000 --strong_model vitb8_dino --num_heads 100 --num_labels 500 --strong_lr 1e-3 --seed 1199 --retrain_stgt --stgt_num_heads 1 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 5000 --strong_model vitb8_dino --num_heads 100 --num_labels 750 --strong_lr 1e-3 --seed 1199 --retrain_stgt --stgt_num_heads 1 --debug
# python run_weak_to_strong.py --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_epochs 5000 --strong_model vitb8_dino --num_heads 100 --strong_lr 1e-3 --seed 1199 --retrain_stgt --stgt_num_heads 1 --debug


# Forward experiments
# Debug
# python run_weak_to_strong.py --dataset cifar10 --weak_model alexnet --retrain_weak --strong_model resnet50_dino --seed 6281 --debug --strong_epochs 1 --weak_epochs 1 --forward


# Resnet50 Cifar 10 Forward, nh = 100
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward --retrain_stgt --retrain_weak
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6496 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 268  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward

# nh =1
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6281 --num_heads 1 --stgt_num_heads 1 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward --retrain_stgt
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 6496 --num_heads 1 --stgt_num_heads 1 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model resnet50_dino --seed 268  --num_heads 1 --stgt_num_heads 1 --stgt_epochs 200 --strong_epochs 100 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward

# Vitb8 Cifar 10 Forward, nh=100
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward --retrain_stgt
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6496 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 268  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward

# # nh =1
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6281 --num_heads 1 --stgt_num_heads 1 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward --retrain_stgt
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6496 --num_heads 1 --stgt_num_heads 1 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 268  --num_heads 1 --stgt_num_heads 1 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward

python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6281 --num_heads 50 --stgt_num_heads 50 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward --retrain_stgt
python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 6496 --num_heads 50 --stgt_num_heads 50 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward
python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset cifar10 --weak_model alexnet --strong_model vitb8_dino    --seed 268  --num_heads 50 --stgt_num_heads 50 --stgt_epochs 30 --strong_epochs 35 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --forward

# Imagenet forward resnet50
# nh = 100
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model resnet50_dino  --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --retrain_stgt --retrain_weak --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 6496 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 268  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --forward

# # nh = 1
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model resnet50_dino  --seed 6281 --num_heads 1 --stgt_num_heads 1 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --retrain_stgt --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 6496 --num_heads 1 --stgt_num_heads 1 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model resnet50_dino    --seed 268  --num_heads 1 --stgt_num_heads 1 --stgt_epochs 50 --strong_epochs 150 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --forward

# Imagenet forward vitb8
# nh = 100
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6281 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --retrain_stgt --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6496 --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 268  --num_heads 100 --stgt_num_heads 100 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --forward

# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6281 --num_heads 1 --stgt_num_heads 1 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --retrain_stgt --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6496 --num_heads 1 --stgt_num_heads 1 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --forward
# python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 268  --num_heads 1 --stgt_num_heads 1 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --forward

python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6281 --num_heads 50 --stgt_num_heads 50 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --retrain_stgt --forward
python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 6496 --num_heads 50 --stgt_num_heads 50 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --forward
python run_weak_to_strong.py --exp_id w-to-s-forward-2 --dataset imagenet --weak_model alexnet --strong_model vitb8_dino    --seed 268  --num_heads 50 --stgt_num_heads 50 --stgt_epochs 15 --strong_epochs 40 --strong_batch_size 256 --weak_validation_split 0.2 --weak_split 0.3 --optimizer adamw --strong_weight_decay 0.1 --use_alexnet_classifier --forward

# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --retrain_stgt --strong_model vitb8_dino --seed 6281 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 6496 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35
# python run_weak_to_strong.py --exp_id w-to-s --dataset imagenet --weak_model alexnet --use_alexnet_classifier --strong_model vitb8_dino --seed 268 --num_heads 100 --stgt_num_heads 100 --strong_epochs 35

# More seeds
# 10902
# 10054
# 1963
# 4921
# 9977
# 5513
# 126
# 6334
# 7698
# 7540
# 7665
# 10722
# 5573
# 7886
# 10259
# 9004
# 9249
# 10761
# 10568
# 10314
