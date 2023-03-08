import os

from PIL import Image
import numpy as np



def load_images(fig_dir, seed, tag, iter_tag, config_names):
    images = []
    for config_name in config_names:
        fig_path = os.path.join(fig_dir, config_name, tag)
        fig_name = seed + '_' + tag + f'{iter_tag}.png'
        fig_path = os.path.join(fig_path, fig_name)
        print(fig_path)
        images.append(Image.open(fig_path))

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
        "channels_fixall.png":[
            "cifar10_32_100_hybrid_conv_128s16k3l12 cifar10_32_100_hybrid_conv_256s16k3l12 cifar10_32_100_hybrid_conv_512s16k3l12" + \
            " cifar10_32_100_hybrid_conv_128s16k7l12 cifar10_32_100_hybrid_conv_256s16k7l12", 
            (2,3)
        ],
        "sizes_fixall.png":[
            "cifar10_32_100_hybrid_conv_512s16k3l12 cifar10_32_100_hybrid_conv_512s8k3l12 cifar10_32_100_hybrid_conv_512s4k1l12", 
            (1,3)
        ],
        "layers_fixall.png":[
            "cifar10_32_100_hybrid_conv_512s4k1l9 cifar10_32_100_hybrid_conv_512s4k1l12 cifar10_32_100_hybrid_conv_512s4k1l15" + \
            " cifar10_32_100_hybrid_conv_512s4k3l9 cifar10_32_100_hybrid_conv_512s4k3l12 cifar10_32_100_hybrid_conv_512s4k3l15",
            (2,3)
        ],
        "downsizes_fixchennel128.png":[
            "cifar10_32_100_hybrid_conv_128down2k3l12 cifar10_32_100_hybrid_conv_128down4k3l9 cifar10_32_100_hybrid_conv_128down8k3l6" + \
            " cifar10_32_100_hybrid_conv_128down2k3l24 cifar10_32_100_hybrid_conv_128down4k3l18 cifar10_32_100_hybrid_conv_128down8k3l12", 
            (2,3)
        ],
        "downsizes_fixchennel512.png":[
            "cifar10_32_100_hybrid_conv_512down2k3l12 cifar10_32_100_hybrid_conv_512down4k3l9 cifar10_32_100_hybrid_conv_512down8k3l6" + \
            " cifar10_32_100_hybrid_conv_512down2k3l24 cifar10_32_100_hybrid_conv_512down4k3l18 cifar10_32_100_hybrid_conv_512down8k3l12", 
            (2,3)
        ],
        "downsizes_difflayers_fixchennel512.png":[
            "cifar10_32_100_hybrid_conv_512down2k3l12 cifar10_32_100_hybrid_conv_512down2k3l18_3663 cifar10_32_100_hybrid_conv_512down2k3l18_6336" + \
            " cifar10_32_100_hybrid_conv_512down2k3l18_6363 cifar10_32_100_hybrid_conv_512down2k3l18_6633 cifar10_32_100_hybrid_conv_512down2k3l24", 
            (2,3)
        ],
        "downsize_upchannel.png":[
            "cifar10_32_100_hybrid_conv_up256down2k3l12 cifar10_32_100_hybrid_conv_up512down2k3l12  cifar10_32_100_hybrid_conv_up1024down2k3l12" + \
            " cifar10_32_100_hybrid_conv_up256down4k3l9 cifar10_32_100_hybrid_conv_up512down4k3l9 cifar10_32_100_hybrid_conv_up1024down4k3l9", 
            (2,3)
        ],
        "difflayers_downsize_upchannel512.png":[
            "cifar10_32_100_hybrid_conv_up512down2k3l12 cifar10_32_100_hybrid_conv_up512down2k3l15_m256" + \
            " cifar10_32_100_hybrid_conv_up512down2k3l15_m512 cifar10_32_100_hybrid_conv_up512down2k3l18_m256_m512", 
            (2,2)
        ],
        "difflayers_fc_downsize_upchannel128.png":[
            "cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf3_fc2048 cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf9_fc2048" + \
            " cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf3_fc2048_bn cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf6_fc2048_bn cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf9_fc2048_bn", 
            (2,3)
        ],
        "difflayers_fc_downsize_upchannel512.png":[
            "cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf6_fc2048 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf9_fc2048" + \
            " cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048_bn cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf6_fc2048_bn cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf9_fc2048_bn", 
            (2,3)
        ],
        "channels_fc_downsize_upchannel.png":[
            "cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf3_fc2048 cifar10_32_100_hybrid_conv_fc_up256down4k3l9lf6_fc1024 cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048" + \
            " cifar10_32_100_hybrid_conv_fc_up128down4k3l9lf3_fc2048_bn cifar10_32_100_hybrid_conv_fc_up256down4k3l9lf6_fc1024_bn cifar10_32_100_hybrid_conv_fc_up512down4k3l9lf3_fc2048_bn", 
            (2,3)
        ],
        "layers_fc_fixchannel2048.png":[
            "cifar10_32_100_hybrid_fc_l9_fc2048 cifar10_32_100_hybrid_fc_l12_fc2048 cifar10_32_100_hybrid_fc_l15_fc2048" + \
            " cifar10_32_100_hybrid_fc_l6_fc2048_bn cifar10_32_100_hybrid_fc_l9_fc2048_bn cifar10_32_100_hybrid_fc_l12_fc2048_bn", 
            (2,3)
        ],
    }

    save_name = "downsizes_difflayers_fixchennel512.png"
    
    config_names = EXPRS_DICT[save_name][0]
    config_names = config_names.split()

    # Hyper parameters
    fig_dir = './figures/conv_fc_comparison'
    seed = '177'
    tag = 'new-block-in-down-rand'
    iter_tag = '_iter3500'
    shape = EXPRS_DICT[save_name][1]
    size = 512

    images = load_images(fig_dir, seed, tag, iter_tag, config_names)
    # resize
    c = 0
    resized_images = []
    for c in range(len(images)):
        tmp_img = images[c]
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
            whole_image[rh*i:rh*(i+1), rw*j:rw*(j+1), :] = tmp_img
            c += 1
    save_path = os.path.join(fig_dir, 'results', save_name)
    Image.fromarray(whole_image.astype(np.uint8)).save(save_path)

            



            






