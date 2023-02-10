# config_name=cifar10_32_100_inceptionv3
# seed=197

# config_names="cifar10_32_100_inceptionv3 cifar10_32_100_resnet34 cifar10_32_100_resnet50 cifar10_32_100_resnet101 cifar10_32_100_vgg16 cifar10_32_100_vgg19"
# seeds="463 759 777 919 7 99"
# 
#[] 
config_names="cifar10_32_100_hybrid_v1"
# seeds="97 177 197 223 337 463 759 777 919 7 99 103"
seeds="97 177 197"
# seeds="97"
for config_name in $config_names
do
    for seed in $seeds
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
                               --gpu-ids 0 \
                               --seed $seed
    done
done