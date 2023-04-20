import os

from PIL import Image
import numpy as np



def load_images(fig_dir, seed, tag, metric_tag, iter_tag, config_names):
    images = []
    for config_name in config_names:
        fig_path = os.path.join(fig_dir, config_name, tag)
        fig_name = seed + '_' + tag + f'{metric_tag}{iter_tag}.png'
        fig_path = os.path.join(fig_path, fig_name)
        print(fig_path)
        if os.path.exists(fig_path):
            images.append(Image.open(fig_path))
        else:
            images.append(None)

    return images

        
if __name__ == '__main__':
    # results that need to be combined or drawn
    # Structure:Same channel and Spatial Size
    # channels: 128-256-512
    # config_names = "cifar10_32_100_hybrid_conv_128s16k3l12 cifar10_32_100_hybrid_conv_256s16k3l12 cifar10_32_100_hybrid_conv_512s16k3l12" + \
    #             " cifar10_32_100_hybrid_conv_128s16k7l12 cifar10_32_100_hybrid_conv_256s16k7l12"

    # Structure:Same channel and Spatial Size
    # spatial: 16-4
    # config_names = "cifar10_32_100_hybrid_conv_512s16k3l12 cifar10_32_100_hybrid_conv_512s8k3l12 cifar10_32_100_hybrid_conv_512s4k1l12"

    EXPRS_DICT = {
        # name meaning: the term 1 want to compare + the term 2 want to compare + structure info.
        # Fix all channels and spatial sizes.
        "kernels_channels_fixall.png":[
            "cifar10_32_100_hybrid_conv_128s16k1l12 cifar10_32_100_hybrid_conv_256s16k1l12 cifar10_32_100_hybrid_conv_512s16k1l12" + \
            " cifar10_32_100_hybrid_conv_128s16k3l12 cifar10_32_100_hybrid_conv_256s16k3l12 cifar10_32_100_hybrid_conv_512s16k3l12" + \
            " cifar10_32_100_hybrid_conv_128s16k7l12 cifar10_32_100_hybrid_conv_256s16k7l12 cifar10_32_100_hybrid_conv_512s16k7l12", 
            (3, 3)
        ],

        "channels_sizes_fixall.png":[
            "cifar10_32_100_hybrid_conv_128s4k3l12 cifar10_32_100_hybrid_conv_128s8k3l12 cifar10_32_100_hybrid_conv_128s16k3l12" + \
            " cifar10_32_100_hybrid_conv_512s4k3l12 cifar10_32_100_hybrid_conv_512s8k3l12 cifar10_32_100_hybrid_conv_512s16k3l12", 
            (2, 3)
        ],

        "sizes_kernels_fixall.png":[
            "cifar10_32_100_hybrid_conv_128s4k1l12 cifar10_32_100_hybrid_conv_128s4k3l12 cifar10_32_100_hybrid_conv_128s4k5l12 cifar10_32_100_hybrid_conv_128s4k7l12" + \
            " cifar10_32_100_hybrid_conv_128s8k1l12 cifar10_32_100_hybrid_conv_128s8k3l12 cifar10_32_100_hybrid_conv_128s8k5l12 cifar10_32_100_hybrid_conv_128s8k7l12" + \
            " cifar10_32_100_hybrid_conv_128s16k1l12 cifar10_32_100_hybrid_conv_128s16k3l12 cifar10_32_100_hybrid_conv_128s16k5l12 cifar10_32_100_hybrid_conv_128s16k7l12", 
            (3, 4)
        ],

        'sizes_kernels_fixall_norelu.png':[
            "cifar10_32_100_hybrid_conv_128s4k1l12_norelu cifar10_32_100_hybrid_conv_128s4k3l12_norelu cifar10_32_100_hybrid_conv_128s4k7l12_norelu" + \
            " cifar10_32_100_hybrid_conv_128s8k1l12_norelu cifar10_32_100_hybrid_conv_128s8k3l12_norelu cifar10_32_100_hybrid_conv_128s8k7l12_norelu" + \
            " cifar10_32_100_hybrid_conv_128s16k1l12_norelu cifar10_32_100_hybrid_conv_128s16k3l12_norelu cifar10_32_100_hybrid_conv_128s16k7l12_norelu", 
            (3, 3)
        ],
        "sizes_kernels_fixall_l36.png":[
            "cifar10_32_100_hybrid_conv_128s4k1l36 cifar10_32_100_hybrid_conv_128s4k3l36 cifar10_32_100_hybrid_conv_128s4k7l36" + \
            " cifar10_32_100_hybrid_conv_128s16k1l36 cifar10_32_100_hybrid_conv_128s16k3l36 cifar10_32_100_hybrid_conv_128s16k7l36", 
            (2, 3)
        ],
        'sizes_kernels_fixall_l36_norelu.png':[
            "cifar10_32_100_hybrid_conv_128s4k1l36_norelu cifar10_32_100_hybrid_conv_128s4k3l36_norelu cifar10_32_100_hybrid_conv_128s4k7l36_norelu" + \
            " cifar10_32_100_hybrid_conv_128s16k1l36_norelu cifar10_32_100_hybrid_conv_128s16k3l36_norelu cifar10_32_100_hybrid_conv_128s16k7l36_norelu", 
            (2, 3)
        ],
        "layers_kernels_fixall.png":[
            "cifar10_32_100_hybrid_conv_512s4k1l9 cifar10_32_100_hybrid_conv_512s4k1l12 cifar10_32_100_hybrid_conv_512s4k1l15" + \
            " cifar10_32_100_hybrid_conv_512s4k3l9 cifar10_32_100_hybrid_conv_512s4k3l12 cifar10_32_100_hybrid_conv_512s4k3l15",
            (2, 3)
        ],

        'sizes_kernels_fixall_channel512_norelu.png':[
            "cifar10_32_100_hybrid_conv_512s4k1l12_norelu cifar10_32_100_hybrid_conv_512s4k3l12_norelu cifar10_32_100_hybrid_conv_512s4k7l12_norelu" + \
            " cifar10_32_100_hybrid_conv_512s16k1l12_norelu cifar10_32_100_hybrid_conv_512s16k3l12_norelu cifar10_32_100_hybrid_conv_512s16k7l12_norelu", 
            (2, 3)
        ],
        'sizes_kernels_fixall_channel512_l36_norelu.png':[
            "cifar10_32_100_hybrid_conv_512s16k1l36_norelu cifar10_32_100_hybrid_conv_512s16k3l36_norelu", 
            (1, 2)
        ],
        ######################################################################################################################################################################################
        # downsizes fixchannels
        "layers_sizes_downsizes_fixchennel128.png":[
            "cifar10_32_100_hybrid_conv_128down2k3l12 cifar10_32_100_hybrid_conv_128down4k3l9 cifar10_32_100_hybrid_conv_128down8k3l6" + \
            " cifar10_32_100_hybrid_conv_128down2k3l24 cifar10_32_100_hybrid_conv_128down4k3l18 cifar10_32_100_hybrid_conv_128down8k3l12" + \
            " cifar10_32_100_hybrid_conv_128down2k3l48 cifar10_32_100_hybrid_conv_128down4k3l36", 
            (3, 3)
        ],

        "layers_sizes_downsizes_fixchennel512.png":[
            "cifar10_32_100_hybrid_conv_512down2k3l12 cifar10_32_100_hybrid_conv_512down4k3l9 cifar10_32_100_hybrid_conv_512down8k3l6" + \
            " cifar10_32_100_hybrid_conv_512down2k3l24 cifar10_32_100_hybrid_conv_512down4k3l18 cifar10_32_100_hybrid_conv_512down8k3l12" +  
            " cifar10_32_100_hybrid_conv_512down2k3l48 cifar10_32_100_hybrid_conv_512down4k3l36 cifar10_32_100_hybrid_conv_512down8k3l6",
            (3, 3)
        ],

        "difflayers_downsizes_fixchennel512_l18.png":[
            "cifar10_32_100_hybrid_conv_512down2k3l18_3663 cifar10_32_100_hybrid_conv_512down2k3l18_6336 cifar10_32_100_hybrid_conv_512down2k3l18_6363 cifar10_32_100_hybrid_conv_512down2k3l18_6633",
            (1, 4)
        ],

        "difflayers_downsizes_fixchennel512_l36.png":[
            "cifar10_32_100_hybrid_conv_512down2k3l18_6CC6 cifar10_32_100_hybrid_conv_512down2k3l18_C66C cifar10_32_100_hybrid_conv_512down2k3l18_C6C6 cifar10_32_100_hybrid_conv_512down2k3l18_CC66",
            (1, 4)
        ],

        "shuffle_downsizes_fixchennel128.png":[
            "cifar10_32_100_hybrid_conv_128down2k3l12 cifar10_32_100_hybrid_conv_128down2Ash284k3l12 cifar10_32_100_hybrid_conv_128down2Ash428k3l12" + \
            " cifar10_32_100_hybrid_conv_128down2Ash482k3l12 cifar10_32_100_hybrid_conv_128down2Ash824k3l12 cifar10_32_100_hybrid_conv_128down2Ash248k3l12",
            (2, 3)
        ],

        ######################################################################################################################################################################################
        # fixsizes upchannels
        "channels_sizes_layers_fixsize_upchannel.png":[
            "cifar10_32_100_hybrid_conv_up128s4Ak3l12 cifar10_32_100_hybrid_conv_up128s8Ak3l12 cifar10_32_100_hybrid_conv_up128s16Ak3l12" + \
            " cifar10_32_100_hybrid_conv_up512s4Ak3l12 cifar10_32_100_hybrid_conv_up512s8Ak3l12 cifar10_32_100_hybrid_conv_up512s16Ak3l12" + \
            " cifar10_32_100_hybrid_conv_up128s4Ak3l36 cifar10_32_100_hybrid_conv_up128s8Ak3l36 cifar10_32_100_hybrid_conv_up128s16Ak3l36" + \
            " cifar10_32_100_hybrid_conv_up512s4Ak3l36 cifar10_32_100_hybrid_conv_up512s8Ak3l36 cifar10_32_100_hybrid_conv_up512s16Ak3l36",
            (4, 3)
        ],
        "shufflechannel_fixsize_upchannel_size4.png":[
            "cifar10_32_100_hybrid_conv_up128s4Ak3l12 cifar10_32_100_hybrid_conv_up128sh136s4Ak3l12 cifar10_32_100_hybrid_conv_up128sh163s4Ak3l12" + \
            " cifar10_32_100_hybrid_conv_up128sh316s4Ak3l12 cifar10_32_100_hybrid_conv_up128sh613s4Ak3l12 cifar10_32_100_hybrid_conv_up128sh631s4Ak3l12",
            (2, 3)
        ],
        "shufflechannel_fixsize_upchannel_size16.png":[
            "cifar10_32_100_hybrid_conv_up128s16Ak3l12 cifar10_32_100_hybrid_conv_up128sh136s16Ak3l12 cifar10_32_100_hybrid_conv_up128sh163s16Ak3l12" + \
            " cifar10_32_100_hybrid_conv_up128sh316s16Ak3l12 cifar10_32_100_hybrid_conv_up128sh613s16Ak3l12 cifar10_32_100_hybrid_conv_up128sh631s16Ak3l12",
            (2, 3)
        ],

        'channels_sizes_layers_fixsize_upchannel_norelu.png':[
            "cifar10_32_100_hybrid_conv_up128s4Ak3l12_norelu cifar10_32_100_hybrid_conv_up128s16Ak3l12_norelu cifar10_32_100_hybrid_conv_up128s4Ak3l36_norelu cifar10_32_100_hybrid_conv_up128s16Ak3l36_norelu" + \
            " cifar10_32_100_hybrid_conv_up512s4Ak3l12_norelu cifar10_32_100_hybrid_conv_up512s8Ak3l12_norelu cifar10_32_100_hybrid_conv_up512s4Ak3l36_norelu cifar10_32_100_hybrid_conv_up512s16Ak3l36_norelu",
            (2, 4)
        ],
        ######################################################################################################################################################################################
        # downsizes upchannels
        "sizes_channels_downsize_upchannel.png":[
            "cifar10_32_100_hybrid_conv_up256down2k3l12 cifar10_32_100_hybrid_conv_up512down2k3l12 cifar10_32_100_hybrid_conv_up1024down2k3l12" + \
            " cifar10_32_100_hybrid_conv_up256down4k3l9 cifar10_32_100_hybrid_conv_up512down4k3l9 cifar10_32_100_hybrid_conv_up1024down4k3l9", 
            (2, 3)
        ],
        "difflayers_downsize_upchannel512.png":[
            "cifar10_32_100_hybrid_conv_up512down2k3l12 cifar10_32_100_hybrid_conv_up512down2k3l15_m256" + \
            " cifar10_32_100_hybrid_conv_up512down2k3l15_m512 cifar10_32_100_hybrid_conv_up512down2k3l18_m256_m512", 
            (2, 2)
        ],

        "shufflechannel_downsize_upchannel512.png":[
            "cifar10_32_100_hybrid_conv_up512down2k3l12 cifar10_32_100_hybrid_conv_up512sh152down2Ak3l12 cifar10_32_100_hybrid_conv_up512sh215down2Ak3l12" + \
            " cifar10_32_100_hybrid_conv_up512sh251down2Ak3l12 cifar10_32_100_hybrid_conv_up512sh512down2Ak3l12 cifar10_32_100_hybrid_conv_up512sh521down2Ak3l12", 
            (2, 3)
        ],
        
        "shufflesizes_downsize_upchannel512.png":[
            "cifar10_32_100_hybrid_conv_up512down2k3l12 cifar10_32_100_hybrid_conv_up512down2Ash248k3l12 cifar10_32_100_hybrid_conv_up512down2Ash284k3l12" + \
            " cifar10_32_100_hybrid_conv_up512down2Ash428k3l12 cifar10_32_100_hybrid_conv_up512down2Ash482k3l12 cifar10_32_100_hybrid_conv_up512down2Ash824k3l12", 
            (2, 3)
        ],

        ######################################################################################################################################################################################
        # downsizes upchannels fc

        "difflayers_fc_downsize_upchannel128.png":[
            "cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf3_fc2048 cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf9_fc2048" + \
            " cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf3_fc2048_bn cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf6_fc2048_bn cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf9_fc2048_bn", 
            (2, 3)
        ],
        "difflayers_fc_downsize_upchannel512.png":[
            "cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf9_fc2048" + \
            " cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048_bn cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf6_fc2048_bn cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf9_fc2048_bn", 
            (2, 3)
        ],
        "channels_fc_downsize_upchannel.png":[
            "cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf3_fc2048 cifar10_32_100_hybrid_conv_fc_up256down4k3l9lf6_fc1024 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048" + \
            " cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf3_fc2048_bn cifar10_32_100_hybrid_conv_fc_up256down4k3l9lf6_fc1024_bn cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048_bn", 
            (2, 3)
        ],


        'channels_layers_downsize_upchannel_norelu.png':[
            "cifar10_32_100_hybrid_conv_up256down2k3l12_norelu cifar10_32_100_hybrid_conv_up256down2k3l36_norelu" + \
            " cifar10_32_100_hybrid_conv_up512down2k3l12_norelu cifar10_32_100_hybrid_conv_up512down2k3l36_norelu",
            (2, 2)
        ],
        ######################################################################################################################################################################################
        # FC

        'layers_fc_fixchannel2048.png':[
            "cifar10_32_100_hybrid_fc_l9_fc2048 cifar10_32_100_hybrid_fc_l12_fc2048 cifar10_32_100_hybrid_fc_l15_fc2048" + \
            " cifar10_32_100_hybrid_fc_l6_fc2048_bn cifar10_32_100_hybrid_fc_l9_fc2048_bn cifar10_32_100_hybrid_fc_l12_fc2048_bn", 
            (2, 3)
        ],



        # Measures Test
        # Norm Dist
        'sizes_kernels_c128':[
            "cifar10_32_100_phybrid_conv_128s4k3l12A_adam cifar10_32_100_phybrid_conv_128s8k3l12A_adam cifar10_32_100_phybrid_conv_128s16k3l12A_adam " +  
            " cifar10_32_100_phybrid_conv_128s4k5l12A_adam cifar10_32_100_phybrid_conv_128s8k5l12A_adam cifar10_32_100_phybrid_conv_128s16k5l12A_adam" + 
            " cifar10_32_100_phybrid_conv_128s4k7l12A_adam cifar10_32_100_phybrid_conv_128s8k7l12A_adam cifar10_32_100_phybrid_conv_128s16k7l12A_adam", 
            (3, 3)
        ],

        'sizes_kernels_c32':[
            "cifar10_32_100_phybrid_conv_32s4k3l12A_adam cifar10_32_100_phybrid_conv_32s8k3l12A_adam cifar10_32_100_phybrid_conv_32s16k3l12A_adam " +  
            " cifar10_32_100_phybrid_conv_32s4k5l12A_adam cifar10_32_100_phybrid_conv_32s8k5l12A_adam cifar10_32_100_phybrid_conv_32s16k5l12A_adam" + 
            " cifar10_32_100_phybrid_conv_32s4k7l12A_adam cifar10_32_100_phybrid_conv_32s8k7l12A_adam cifar10_32_100_phybrid_conv_32s16k7l12A_adam", 
            (3, 3)
        ],
        'channels_kernels_c32':[
            "cifar10_32_100_phybrid_conv_16s8k3l12A_adam cifar10_32_100_phybrid_conv_32s8k3l12A_adam cifar10_32_100_phybrid_conv_64s8k3l12A_adam" +  
            " cifar10_32_100_phybrid_conv_16s8k5l12A_adam cifar10_32_100_phybrid_conv_32s8k5l12A_adam cifar10_32_100_phybrid_conv_64s8k5l12A_adam" + 
            " cifar10_32_100_phybrid_conv_16s8k7l12A_adam cifar10_32_100_phybrid_conv_32s8k7l12A_adam cifar10_32_100_phybrid_conv_64s8k7l12A_adam", 
            (3, 3)
        ]
        # Sep loss ?
    }

    
    comparison_settings = 'channels_kernels_c32'
    metric_tag = '_proj_vtm_root2'
    # tag = 'conv-norm_dist-norm2-block-in-down'
    tag = 'conv-proj_dist-norm2-fro-mean-adam0.05-pytorch-E50-block-in-down'
    iter_tag = '_iter2500'
    save_name = f'{comparison_settings}{metric_tag}.png'
    
    # Hyper parameters
    fig_dir = './figures/pure_hybrids'
    seed = '97'
    
    size = 512

    config_names = EXPRS_DICT[comparison_settings][0]
    config_names = config_names.split()

    shape = EXPRS_DICT[comparison_settings][1]
    images = load_images(fig_dir, seed, tag, metric_tag, iter_tag, config_names)
    # resize
    c = 0
    resized_images = []
    for c in range(len(images)):
        tmp_img = images[c]
        if tmp_img is None:
            resized_images.append(None)
            continue
        w, h = tmp_img.size
        r = size / w
        rw, rh = int(r * w), int(r * h)
        tmp_img = tmp_img.resize((rw, rh))
        resized_images.append(np.array(tmp_img))

    
    whole_image = np.zeros((rh*shape[0], rw*shape[1], 4))

    c = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if c >= len(resized_images):
                break
            tmp_img = resized_images[c]
            c += 1
            if tmp_img is None:
                continue
            whole_image[rh*i:rh*(i+1), rw*j:rw*(j+1), :] = tmp_img
            
    save_path = os.path.join(fig_dir, 'results', save_name)
    Image.fromarray(whole_image.astype(np.uint8)).save(save_path)

            



            






