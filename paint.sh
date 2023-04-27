
# expr_prefix="conv_fc_comparison"
expr_prefix="pure_hybrids"  # pure_hybrids, measure_test

config_names=${config_names}" cifar10_32_100_phybrid_fc_3072l9_bn"
seeds="97 177"
metric_type="proj_dist"  # margin, sep_loss, sep_stn, sep_stn_loss, stn, proj_dist, norm_dist
plot_metric_type="proj_vtm"
# metric_trans_types="exproot0.5 exproot0.6 exproot0.7 exproot0.8 exproot0.9"  # exp , exproot4
# metric_trans_types=${metric_trans_types}" exproot2.2 exproot2.4 exproot2.6 exproot2.8 exproot3"  # exp , exproot4
# metric_trans_types=${metric_trans_types}" exproot3.2 exproot3.4 exproot3.6 exproot3.8"  # exp , exproot4
# metric_trans_types="exproot1 exproot1.2 exproot1.4 exproot1.6 exproot1.8 exproot2"  # exp , exproot4
# Flexible
# metric_trans_types=${metric_trans_types}" exproot0.6 exproot0.8 exproot1 exproot1.2 exproot1.4 exproot1.6"  # exp , exproot4
# metric_trans_types="exproot2"
# metric_trans_types="exproot8 exproot10"
# metric_trans_types="exproot0.1 exproot0.2 exproot0.3 exproot0.4"
# metric_trans_types="root6 root8 root10 root12"
# metric_trans_types="root16 root18 root20 log"
metric_trans_types="root2"
add_metric_types="proj_vtm"
# tag='conv-norm_dist-norm2-block-in-down'
# tag='conv-norm_dist-norm2-mean-block-in-down'
# conv-proj_dist-normfro-fro-mean-adam0.05-pytorch-E5-block-in-down
# tag='conv-proj_dist-norm2-fro-mean-adam0.05-pytorch-E50-block-in-down-aug'
# tag='conv-proj_dist-norm2-fro-mean-adam0.05-pytorch-E50-block-in-down'
tag='conv-proj_dist-normfro-fro-mean-adam0.05-pytorch-E5-block-in-down'
# tag="conv-stn-block-in-down-rand"
# tag='conv-gstn-dim2-block-in-down-rand'
# tag='conv-norm_dist-normfro-block-in-down'
# tag="new-block-in-down-rand"
# tag="conv-sep_stn_loss-C64-E500-downfc-adam0.1-block-in-down"
# tag="conv-sep_stn_loss-C16-E300-adam0.1-block-in-down"
for seed in $seeds
do
    for metric_trans_type in $metric_trans_types
    do
        for config_name in $config_names    
        do               
            echo $seed
            echo $config_name
            python3 paint_from_record.py --model_names $config_name \
                                --seeds $seed \
                                --feat_types $tag \
                                --expr_prefix $expr_prefix \
                                --metric_type $metric_type \
                                --plot_metric_type $plot_metric_type \
                                --metric_trans_type $metric_trans_type \
                                --add_metric_types $add_metric_types 
        done
    done
done