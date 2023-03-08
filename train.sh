
expr_prefix="conv_fc_comparison"
# config_names="cifar10_32_100_hybrid_conv_256s16k3l12 cifar10_32_100_hybrid_conv_256s16k7l12 cifar10_32_100_hybrid_conv_512s4k1l9 cifar10_32_100_hybrid_conv_512s4k1l15"
# config_names2="cifar10_32_100_hybrid_conv_512s4k3l9 cifar10_32_100_hybrid_conv_512s4k3l15 cifar10_32_100_hybrid_conv_512s8k3l12 cifar10_32_100_hybrid_conv_512s16k3l12 cifar10_32_100_hybrid_conv_512s16k7l12"
# config_names="cifar10_32_100_hybrid_conv_512s16k7l12"

# config_names="cifar10_32_100_hybrid_conv_512s16k7l12 cifar10_32_100_hybrid_conv_up1024down2k3l12 cifar10_32_100_hybrid_conv_up1024down4k3l9 cifar10_32_100_hybrid_conv_512down2k3l24 cifar10_32_100_hybrid_conv_512down4k3l18 cifar10_32_100_hybrid_conv_512down8k3l6 cifar10_32_100_hybrid_conv_512down8k3l12"
# config_names=${config_names}" cifar10_32_100_hybrid_conv_128down2k3l24 cifar10_32_100_hybrid_conv_128down4k3l18 cifar10_32_100_hybrid_conv_128down8k3l6 cifar10_32_100_hybrid_conv_128down8k3l12"
# config_names=${config_names}" cifar10_32_100_hybrid_conv_512down2k3l18_3663 cifar10_32_100_hybrid_conv_512down2k3l18_6336 cifar10_32_100_hybrid_conv_512down2k3l18_6363 cifar10_32_100_hybrid_conv_512down2k3l18_6633"
# config_names=${config_names}" cifar10_32_100_hybrid_fc_l9_fc2048 cifar10_32_100_hybrid_fc_l12_fc2048 cifar10_32_100_hybrid_fc_l15_fc2048"
# config_names=${config_names}" cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf9_fc2048 cifar10_32_100_hybrid_conv_fc_up2048down4k3l9lf6_fc2048"
# config_names=${config_names}" cifar10_32_100_hybrid_fc_l6_fc2048_bn cifar10_32_100_hybrid_fc_l9_fc2048_bn cifar10_32_100_hybrid_fc_l12_fc2048_bn"

# config_names="cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf3_fc2048 cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf3_fc2048_bn cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf6_fc2048_bn"
# config_names=${config_names}" cifar10_32_100_hybrid_fc_l15_fc2048_SGD cifar10_32_100_hybrid_fc_l12_fc2048_SGD cifar10_32_100_hybrid_fc_l9_fc2048_SGD"
# config_names=${config_names}" cifar10_32_100_hybrid_conv_512down2k3l18_6CC6 cifar10_32_100_hybrid_conv_512down2k3l18_C6C6 cifar10_32_100_hybrid_conv_512down2k3l18_C66C cifar10_32_100_hybrid_conv_512down2k3l18_CC66"
# config_names=${config_names}" cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf9_fc2048 cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf9_fc2048_bn cifar10_32_100_hybrid_conv_fc_up256down4k3l9lf6_fc1024 cifar10_32_100_hybrid_conv_fc_up256down4k3l9lf6_fc1024_bn"
# config_names=${config_names}" cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048_bn cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf6_fc2048_bn"
# config_names=${config_names}" cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf9_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf9_fc2048_bn cifar10_32_100_hybrid_conv_fc_up2048down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up2048down4k3l9lf6_fc2048_bn"


config_names="cifar10_32_100_hybrid_conv_128s4k1l12 cifar10_32_100_hybrid_conv_128s4k3l12 cifar10_32_100_hybrid_conv_128s4k5l12 cifar10_32_100_hybrid_conv_128s4k7l12"
config_names=${config_names}" cifar10_32_100_hybrid_conv_128s8k1l12 cifar10_32_100_hybrid_conv_128s8k3l12 cifar10_32_100_hybrid_conv_128s8k5l12 cifar10_32_100_hybrid_conv_128s8k7l12"
config_names=${config_names}" cifar10_32_100_hybrid_conv_128s16k1l12 cifar10_32_100_hybrid_conv_128s16k3l12 cifar10_32_100_hybrid_conv_128s16k5l12 cifar10_32_100_hybrid_conv_128s16k7l12"
config_names=${config_names}" cifar10_32_100_hybrid_conv_1024s16k3l12"
# # seeds="97 177 197 223 337 463 759 777 919 7 99 103"
seeds="97 177 197"

gpu_num=2
max_num=20
curr_num=0
run_names=""
for seed in $seeds
do
    for config_name in $config_names    
    do
        date=$seed
        workdir=./exprs/$expr_prefix/$config_name/$date/

        if [[ ! -d $workdir ]]; then
            mkdir $workdir
        fi

        cp ./exprs/$expr_prefix/$config_name/config.yaml $workdir
        
        gpu_id=$(($curr_num % $gpu_num))
        FINAL_FILE=./exprs/$expr_prefix/$config_name/$seed/chckpoints/final.pth
        if [[ ! -f "$FINAL_FILE" ]]
        then
            echo $seed
            echo $config_name
            python3 train.py --config ./exprs/$expr_prefix/$config_name/config.yaml \
                                --work-dir $workdir \
                                --gpu-ids $gpu_id \
                                --seed $seed &
        
            curr_num=$(($curr_num + 1))
            run_names=${run_names}" "${config_name}/${seed}
        fi
        if [[ ${curr_num} -ge ${max_num} ]]
        then
            finished_num=0
            while [[ ${finished_num} == 0 ]]
            do
                sleep 1200
                finished_num=0
                tmp_run_names=""
                for run_name in $run_names
                do
                    FILE=./exprs/$expr_prefix/$run_name/chckpoints/final.pth
                    if [[ -f "$FILE" ]]
                    then
                        finished_num=$(($finished_num + 1))
                        echo "Finshed: "$FILE
                    else
                        tmp_run_names=${tmp_run_names}" "${run_name}
                    fi 
                done
            done
            run_names=${tmp_run_names}
            curr_num=$(($curr_num - $finished_num))
        fi
    done
done

finished_num=0
while [[ ${finished_num} != ${curr_num} ]]
do
    sleep 1200
    finished_num=0
    tmp_run_names=""
    for run_name in $run_names
    do
        FILE=./exprs/$expr_prefix/$run_name/chckpoints/final.pth
        if [[ -f "$FILE" ]]
        then
            finished_num=$(($finished_num + 1))
            echo "Finshed: "$FILE
        else
            tmp_run_names=${tmp_run_names}" "${run_name}
        fi 
    done
done

echo "Finished."

# config_names="cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf9_fc2048 cifar10_32_100_hybrid_conv_fc_up2048down4k3l9lf6_fc2048"

# for seed in $seeds
# do
#     for config_name in $config_names    
#     do
#         echo $seed
#         echo $config_name
#         date=$seed
#         workdir=./exprs/$expr_prefix/$config_name/$date/

#         if [ ! -d $workdir ]; then
#             mkdir $workdir
#         fi

#         cp ./exprs/$expr_prefix/$config_name/config.yaml $workdir
         
#         python3 train.py --config ./exprs/$expr_prefix/$config_name/config.yaml \
#                                --work-dir $workdir \
#                                --gpu-ids 0 \
#                                --seed $seed &
#     done
# done

# config_names="cifar10_32_100_hybrid_conv_up256down2k3l12 cifar10_32_100_hybrid_conv_up256down4k3l9 cifar10_32_100_hybrid_conv_up512down2k3l12 cifar10_32_100_hybrid_conv_up512down2k3l15_m256 cifar10_32_100_hybrid_conv_up512down2k3l15_m512"

# for seed in $seeds
# do
#     for config_name in $config_names    
#     do
#         echo $seed
#         echo $config_name
#         date=$seed
#         workdir=./exprs/$expr_prefix/$config_name/$date/

#         if [ ! -d $workdir ]; then
#             mkdir $workdir
#         fi

#         cp ./exprs/$expr_prefix/$config_name/config.yaml $workdir
         
#         python3 train.py --config ./exprs/$expr_prefix/$config_name/config.yaml \
#                                --work-dir $workdir \
#                                --gpu-ids 1 \
#                                --seed $seed &
#     done
# done

# FILE=./exprs/$expr_prefix/cifar10_32_100_hybrid_conv_fc_up2048down4k3l9lf6_fc2048/197/chckpoints/iter_4500.pth

# while [ ! -f "$FILE" ];
# do
#     sleep 1800
# done


# config_names="cifar10_32_100_hybrid_conv_up512down2k3l18_m256_m512 cifar10_32_100_hybrid_conv_up512down4k3l9 cifar10_32_100_hybrid_conv_up1024down2k3l12 cifar10_32_100_hybrid_conv_up1024down4k3l9"

# for seed in $seeds
# do
#     for config_name in $config_names    
#     do
#         echo $seed
#         echo $config_name
#         date=$seed
#         workdir=./exprs/$expr_prefix/$config_name/$date/

#         if [ ! -d $workdir ]; then
#             mkdir $workdir
#         fi

#         cp ./exprs/$expr_prefix/$config_name/config.yaml $workdir
         
#         python3 train.py --config ./exprs/$expr_prefix/$config_name/config.yaml \
#                                --work-dir $workdir \
#                                --gpu-ids 0 \
#                                --seed $seed &
#     done
# done

# FILE=./exprs/$expr_prefix/cifar10_32_100_hybrid_conv_up512down2k3l18_m256_m512/197/chckpoints/iter_4500.pth

# while [ ! -f "$FILE" ];
# do
#     sleep 1800
# done

