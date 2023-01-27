config_name=cifar10
date=20230127
workdir=./exprs/$config_name/$date/

if [ ! -d $workdir ]; then
    mkdir $workdir
fi

cp ./exprs/$config_name/config.yaml $workdir
 
python3 train.py --config ./exprs/$config_name/config.yaml \
                       --work-dir $workdir \
                       --gpu-ids 0 \
                       --seed 97
