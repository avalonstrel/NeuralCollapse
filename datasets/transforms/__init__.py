# Now just import from pytorch, add custom operation if need

from torchvision.transforms import (Compose, CenterCrop, ColorJitter, Normalize, Pad,
                         RandomCrop, RandomGrayscale, RandomResizedCrop, 
                         RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor)


TRANSFORMS_DICT = {
     'CenterCrop':CenterCrop,
     'ColorJitter':ColorJitter, 
     'Normalize':Normalize, 
     'Pad':Pad,
     'RandomCrop':RandomCrop, 
     'RandomGrayscale':RandomGrayscale, 
     'RandomResizedCrop':RandomResizedCrop, 
     'RandomHorizontalFlip':RandomHorizontalFlip,
     'RandomVerticalFlip':RandomVerticalFlip,
     'Resize':Resize,
     'ToTensor':ToTensor,
}

def build_transforms(transfroms_list, trans_kwargs):
     transforms = []
     for trans_name in transfroms_list:
          if trans_name in trans_kwargs:
               tmp_trans = TRANSFORMS_DICT[trans_name](**trans_kwargs[trans_name])
          else:
               tmp_trans = TRANSFORMS_DICT[trans_name]()
          transforms.append(tmp_trans)
     return transforms


__all__ = [
     'Compose', 'CenterCrop', 'ColorJitter', 'Normalize', 'Pad',
     'RandomCrop', 'RandomGrayscale', 'RandomResizedCrop', 'Resize'
]

