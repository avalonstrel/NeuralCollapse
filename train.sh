# config_name=cifar10_32_100_inceptionv3
# seed=197

# config_names="cifar10_32_100_inceptionv3 cifar10_32_100_resnet34 cifar10_32_100_resnet50 cifar10_32_100_resnet101 cifar10_32_100_vgg16 cifar10_32_100_vgg19"
# seeds="463 759 777 919 7 99"

#[] 
# config_names="cifar10_224_1000_inceptionv3 cifar10_224_1000_vgg13_bn cifar10_224_1000_resnet50_bn"
# config_names="cifar10_256_1000_swin_t cifar10_256_1000_vit_b_16 cifar10_256_1000_hybrid_v4"
# config_names="cifar100_64_100_hybrid_v1 cifar100_32_100_hybrid_v2"
# config_names="cifar100_32_100_hybrid_v1 cifar100_32_100_hybrid_v2 cifar100_32_100_vgg13_bn cifar100_64_100_hybrid_v1"
# config_names="cifar10_32_1000_hybrid_v1 cifar10_32_1000_hybrid_v2 cifar10_32_1000_vgg13_bn"
# config_names="cifar10_32_100_odenet_euler cifar10_32_100_odenet_fixadam"
# config_names="cifar10_32_100_vgg13_bn_nom cifar10_32_100_vgg13_bn_nov"
# config_names="cifar10_224_1000_resnet50_bn cifar10_224_1000_vgg13_bn cifar10_32_1000_vit_b_16"
# config_names="cifar10_256_1000_hybrid_v4 cifar10_256_1000_swin_t cifar100_32_100_vgg13_bn cifar10_32_1000_vgg13_bn"
# config_names="cifar10_64_100_vgg13_bn cifar10_128_100_vgg13_bn cifar10_128_100_hybrid_v2 cifar100_32_100_swin_t cifar100_32_100_vit_b_16"
# config_names="cifar100_224_100_vgg13_bn cifar100_224_100_resnet50_bn cifar100_256_100_swin_t cifar100_256_100_vit_b_16 cifar100_256_100_hybrid_v2 cifar100_256_100_hybrid_v4"
# config_names="cifar10_32_100_odenet_midpoint cifar10_32_100_odenet_rk"
# config_names="cifar100_32_1000_hybrid_v1 cifar100_32_1000_hybrid_v2 cifar100_32_1000_resnet50_bn cifar100_32_1000_swin_t cifar100_32_1000_vgg13_bn cifar100_32_1000_vit_b_16"
# config_names="imagenet_224_10_resnet50_bn imagenet_224_10_vgg13_bn imagenet_256_10_hybrid_v2 imagenet_256_10_hybrid_v4 imagenet_256_10_swin_t imagenet_256_10_vit_b_16"
# config_names="imagenet_224_100_resnet50_bn imagenet_224_100_vgg13_bn imagenet_256_100_hybrid_v2 imagenet_256_100_hybrid_v4 imagenet_256_100_swin_t imagenet_256_100_vit_b_16"
config_names="cifar10_32_6000_hybrid_v1 cifar10_32_6000_hybrid_v2 cifar10_32_6000_resnet50_bn cifar10_32_6000_swin_t cifar10_32_6000_vgg13_bn cifar10_32_6000_vit_b_16"
# cifar10_32_100_vgg13_bn_nov
# seeds="97 177 197 223 337 463 759 777 919 7 99 103"
seeds="97 177 197"
# seeds="97"
for seed in $seeds
do
    for config_name in $config_names    
    do
        echo $seed
        echo $config_name
        date=20230130_$seed
        workdir=./exprs/$config_name/$date/

        if [ ! -d $workdir ]; then
            mkdir $workdir
        fi

        cp ./exprs/$config_name/config.yaml $workdir
         
        python3 train.py --config ./exprs/$config_name/config.yaml \
                               --work-dir $workdir \
                               --gpu-ids 7 \
                               --seed $seed
    done
done