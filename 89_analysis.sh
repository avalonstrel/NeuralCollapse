
expr_prefix="pure_hybrids"  # measure_test pure_hybrids
# config_names=${config_names}" cifar10_32_100_phybrid_conv_32s4k3l12A_adam cifar10_32_100_phybrid_conv_32s4k5l12A_adam cifar10_32_100_phybrid_conv_32s4k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_32s8k3l12A_adam cifar10_32_100_phybrid_conv_32s8k5l12A_adam cifar10_32_100_phybrid_conv_32s8k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_32s16k3l12A_adam cifar10_32_100_phybrid_conv_32s16k5l12A_adam cifar10_32_100_phybrid_conv_32s16k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_128s4k3l12A_adam cifar10_32_100_phybrid_conv_128s4k5l12A_adam cifar10_32_100_phybrid_conv_128s4k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_128s8k3l12A_adam cifar10_32_100_phybrid_conv_128s8k5l12A_adam cifar10_32_100_phybrid_conv_128s8k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_128s16k3l12A_adam cifar10_32_100_phybrid_conv_128s16k5l12A_adam cifar10_32_100_phybrid_conv_128s16k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_128s32k3l12A_adam cifar10_32_100_phybrid_conv_128s32k5l12A_adam cifar10_32_100_phybrid_conv_128s32k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_4s8k3l12A_adam cifar10_32_100_phybrid_conv_4s8k5l12A_adam cifar10_32_100_phybrid_conv_4s8k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_8s8k3l12A_adam cifar10_32_100_phybrid_conv_8s8k5l12A_adam cifar10_32_100_phybrid_conv_8s8k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_256s8k3l12A_adam cifar10_32_100_phybrid_conv_256s8k5l12A_adam cifar10_32_100_phybrid_conv_256s8k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_512s8k3l12A_adam cifar10_32_100_phybrid_conv_512s8k5l12A_adam cifar10_32_100_phybrid_conv_512s8k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_32s4k3l12A_adam cifar10_32_100_phybrid_conv_32s4k5l12A_adam cifar10_32_100_phybrid_conv_32s4k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_32s8k3l12A_adam cifar10_32_100_phybrid_conv_32s8k5l12A_adam cifar10_32_100_phybrid_conv_32s8k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_32s16k3l12A_adam cifar10_32_100_phybrid_conv_32s16k5l12A_adam cifar10_32_100_phybrid_conv_32s16k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_32s32k3l12A_adam cifar10_32_100_phybrid_conv_32s32k5l12A_adam cifar10_32_100_phybrid_conv_32s32k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_64s8k3l12A_adam cifar10_32_100_phybrid_conv_64s8k5l12A_adam cifar10_32_100_phybrid_conv_64s8k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_16s8k3l12A_adam cifar10_32_100_phybrid_conv_16s8k5l12A_adam cifar10_32_100_phybrid_conv_16s8k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_128s32k7l12A_adam_lr00005"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_16s8k3l12A_adam cifar10_32_100_phybrid_conv_16s8k5l12A_adam cifar10_32_100_phybrid_conv_16s8k7l12A_adam"
# config_names=${config_names}" cifar10_32_100_phybrid_conv_64s8k3l12A_adam cifar10_32_100_phybrid_conv_64s8k5l12A_adam cifar10_32_100_phybrid_conv_64s8k7l12A_adam"
config_names=${config_names}" cifar10_32_100_phybrid_fc_3072l9_bn"
# # seeds="97 177 197 223 337 463 759 777 919 7 99 103"
gpu_ids=(0 1)  # (4 5 6 7)  (0 1)
seeds="97 177"
metric_types="norm_dist"  #margin, sep_loss, sep_stn, sep_stn_loss, stn, proj_dist, norm_dist
gstn_dim=2
gpu_num=2
max_num=6
curr_num=0
net_type="fc"
C_intermediates="4"
epoch_num=50
optim_methods="adam" # adam lbfgs
lr=0.05
norm_type=fro
dist_norm_type='fro'
dist_type='mean'
norm_optim_method="pytorch" # pytorch, cvx, cvx_sdp
fig_names=""
datadir="/home/lhy/datasets"
# datadir="/sdc1/hylin/datasets"  
# datadir="/mnt/beegfs/hlinbh/datasets"  

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
                    tag="conv-"${metric_types}"-norm"${norm_type}"-"${dist_norm_type}"-"${dist_type}"-"${optim_method}${lr}"-"${norm_optim_method}"-E"${epoch_num}"-block-in-down-aug"
                elif [ "${metric_types}" = "norm_dist" ]; then
                    tag="conv-"${metric_types}"-norm"${norm_type}"-"${dist_type}"-block-in-down"
                elif [ "${metric_types}" = "gstn" ]; then
                    tag="conv-"${metric_types}"-dim"${gstn_dim}"-svdt-block-in-down-rand"
                else
                    tag="conv-"${metric_types}"-block-in-down-rand"
                fi
                date=$seed
                
                gpu_id=${gpu_ids[$(($curr_num % $gpu_num))]}
                FINAL_FILE=./exprs/$expr_prefix/$config_name/$seed/chckpoints/iter_2500.pth
                FIG_FILE=./figures/$expr_prefix/${config_name}/$tag/${seed}_${tag}.png
                # if [ ! -f "$FINAL_FILE" ]; then
                #     echo "Not exist "$FINAL_FILE
                # fi
                
                # if [ -f "$FIG_FILE" ]; then
                #     echo "Exist "$FIG_FILE
                # fi
                
                # echo $FIG_FILE
                
                if [ -f "$FINAL_FILE" ] && [ ! -f "$FIG_FILE" ];
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
                                        --norm_optim_method $norm_optim_method \
                                        --lr $lr \
                                        --gstn_dim $gstn_dim \
                                        --norm_type $norm_type \
                                        --dist_norm_type $dist_norm_type \
                                        --data-dir $datadir &
                
                    curr_num=$(($curr_num + 1))
                    fig_names=${fig_names}" "${config_name}/$tag/${seed}_${tag}.png
                fi
                if [[ ${curr_num} -ge ${max_num} ]]
                then
                    finished_num=0
                    while [[ ${finished_num} == 0 ]]
                    do
                        sleep 300
                        finished_num=0
                        tmp_fig_names=""
                        for fig_name in $fig_names
                        do
                            FILE=./figures/$expr_prefix/$fig_name
                            if [[ -f "$FILE" ]]
                            then
                                finished_num=$(($finished_num + 1))
                                echo "Finshed: "$FILE
                            else
                                tmp_fig_names=${tmp_fig_names}" "${fig_name}
                            fi 
                        done
                        echo "Current Number:"$curr_num" Finished Number:"$finished_num
                    done
                    fig_names=${tmp_fig_names}
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
    for fig_name in $fig_names
    do
        FILE=./figures/$expr_prefix/$fig_name
        if [[ -f "$FILE" ]]
        then
            finished_num=$(($finished_num + 1))
            echo "Finshed: "$FILE
        fi 
    done
done

echo "Finished."
