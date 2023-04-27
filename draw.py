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
        # Measures Test
        # Norm Dist
        'optims_layers':[
            "cifar10_32_100_phybrid_fc_3072l6_bn_sgd cifar10_32_100_phybrid_fc_3072l9_bn_sgd cifar10_32_100_phybrid_fc_3072l12_bn_sgd cifar10_32_100_phybrid_fc_3072l15_bn_sgd" +  
            " cifar10_32_100_phybrid_fc_3072l6_bn cifar10_32_100_phybrid_fc_3072l9_bn cifar10_32_100_phybrid_fc_3072l12_bn cifar10_32_100_phybrid_fc_3072l15_bn" + 
            " cifar10_32_100_phybrid_fc_3072l6_ln cifar10_32_100_phybrid_fc_3072l9_ln cifar10_32_100_phybrid_fc_3072l12_ln cifar10_32_100_phybrid_fc_3072l15_ln", 
            # " cifar10_32_100_phybrid_fc_3072l6_bn_adam cifar10_32_100_phybrid_fc_3072l9_bn_adam cifar10_32_100_phybrid_fc_3072l12_bn_adam cifar10_32_100_phybrid_fc_3072l15_bn_adam", 
            (3, 4)
        ],
        # Sep loss ?
    }
    
    comparison_settings = 'optims_layers'
    metric_tag = '_stn'
    # tag = 'conv-norm_dist-norm2-block-in-down'
    tag = 'conv-stn-test-block-in-down-rand'
    iter_tag = '_iter2500'
    save_name = f'{comparison_settings}{metric_tag}.png'
    
    # Hyper parameters
    fig_dir = './figures/mlp'
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
    os.makedirs(os.path.join(fig_dir, 'results'), exist_ok=True)
    save_path = os.path.join(fig_dir, 'results', save_name)
    Image.fromarray(whole_image.astype(np.uint8)).save(save_path)

            



            






