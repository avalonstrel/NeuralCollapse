expr_prefix="mlp" 

config_names=${config_names}" cifar10_32_100_phybrid_fc_3072l6_bn cifar10_32_100_phybrid_fc_3072l9_bn cifar10_32_100_phybrid_fc_3072l12_bn cifar10_32_100_phybrid_fc_3072l15_bn"
config_names=${config_names}" cifar10_32_100_phybrid_fc_3072l6_bn_adam cifar10_32_100_phybrid_fc_3072l9_bn_adam cifar10_32_100_phybrid_fc_3072l12_bn_adam cifar10_32_100_phybrid_fc_3072l15_bn_adam"
config_names=${config_names}" cifar10_32_100_phybrid_fc_3072l6_bn_sgd cifar10_32_100_phybrid_fc_3072l9_bn_sgd cifar10_32_100_phybrid_fc_3072l12_bn_sgd cifar10_32_100_phybrid_fc_3072l15_bn_sgd"
config_names=${config_names}" cifar10_32_100_phybrid_fc_3072l6_bnf cifar10_32_100_phybrid_fc_3072l9_bnf cifar10_32_100_phybrid_fc_3072l12_bnf cifar10_32_100_phybrid_fc_3072l15_bnf cifar10_32_100_phybrid_fc_3072l36_bnf"
config_names=${config_names}" cifar10_32_100_phybrid_fc_3072l6_ln cifar10_32_100_phybrid_fc_3072l9_ln cifar10_32_100_phybrid_fc_3072l12_ln cifar10_32_100_phybrid_fc_3072l15_ln cifar10_32_100_phybrid_fc_3072l36_ln"

# the GPU ids that could be used
gpu_ids=(0 1) 
seeds="97 177"
gpu_num=2
max_num=16
curr_num=0
metric_types="stn"

# Ignore following parameters for mlp test  
gstn_dim=2
net_type="downfc"
C_intermediates="16"
epoch_num=5
optim_methods="adam"
lr=0.05
norm_type=2 
dist_norm_type='fro'
dist_type='mean'
norm_optim_method="pytorch"

# the directory that CIFAR10 exist
datadir="/mnt/beegfs/hlinbh/datasets"  

# nvidia-smi 
for seed in $seeds
do
    for C_intermediate in $C_intermediates
    do
        for optim_method in $optim_methods
        do
            for config_name in $config_names    
            do
                if [ "${metric_types}" = "sep_loss" ] | [ "${metric_types}" = "sep_stn" ] | [ "${metric_types}" = "sep_stn_loss" ]; then
                    tag="conv-"${metric_types}"-C"${C_intermediate}"-E"${epoch_num}"-"${net_type}"-"${optim_method}${lr}"-block-in-down"
                elif [ "${metric_types}" = "proj_dist" ]; then
                    tag="conv-"${metric_types}"-norm"${norm_type}"-"${dist_norm_type}"-"${dist_type}"-"${optim_method}${lr}"-"${norm_optim_method}"-E"${epoch_num}"-block-in-down"
                elif [ "${metric_types}" = "norm_dist" ]; then
                    tag="conv-"${metric_types}"-norm"${norm_type}"-"${dist_type}"-d1-block-in-down"
                elif [ "${metric_types}" = "gstn" ]; then
                    tag="conv-"${metric_types}"-dim"${gstn_dim}"-test-block-in-down"
                else
                    tag="conv-"${metric_types}"-test-block-in-down"  #-rand
                fi
                date=$seed
                
                gpu_id=${gpu_ids[$(($curr_num % $gpu_num))]}
                FINAL_FILE=./exprs/$expr_prefix/$config_name/$seed/chckpoints/iter_2500.pth
                # FIG_FILE=./figures/$expr_prefix/${config_name}/$tag/${seed}_${tag}.png
                LOG_FINISHED_FILE=./logs/${config_name}/$tag/${seed}_${tag}_finished.txt

                if [ -f "$FINAL_FILE" ] && [ ! -f "$LOG_FINISHED_FILE" ];
                then
                    echo $seed
                    echo $config_name
                    python3 analysis.py --model_names $config_name \
                                        --gpu_id $gpu_id \
                                        --seeds $seed \
                                        --feat_types $tag \
                                        --expr_prefix $expr_prefix \
                                        --C_intermediate $C_intermediate \
                                        --epoch_num $epoch_num \
                                        --metric_types $metric_types \
                                        --net_type $net_type \
                                        --optim_method $optim_method \
                                        --gstn_dim $gstn_dim \
                                        --norm_optim_method $norm_optim_method \
                                        --lr $lr \
                                        --norm_type $norm_type \
                                        --dist_norm_type $dist_norm_type \
                                        --data-dir $datadir &
                
                    curr_num=$(($curr_num + 1))
                    finished_names=${finished_names}" "${LOG_FINISHED_FILE}
                fi

                # A trick to greedy add programs.
                if [[ ${curr_num} -ge ${max_num} ]]
                then
                    finished_num=0
                    while [[ ${finished_num} == 0 ]]
                    do
                        sleep 300
                        finished_num=0
                        tmp_finished_names=""
                        for finished_name in $finished_names
                        do
                            if [[ -f "$finished_name" ]]
                            then
                                finished_num=$(($finished_num + 1))
                                echo "Finshed: "$finished_name
                            else
                                tmp_finished_names=${tmp_finished_names}" "${finished_name}
                            fi
                        done
                        echo "Current Number:"$curr_num" Finished Number:"$finished_num
                    done
                    finished_names=${tmp_finished_names}
                    curr_num=$(($curr_num - $finished_num))
                    echo "Current Number Update:"$curr_num
                fi
            done
        done
    done
done

finished_num=0
while [[ ${finished_num} != ${curr_num} ]]
do
    sleep 1200
    finished_num=0
    for finished_name in $finished_names
    do
        
        if [[ -f "$finished_name" ]]
        then
            finished_num=$(($finished_num + 1))
            echo "Final Check Finished: "$finished_name
        fi 
    done
done

echo "Finished."
