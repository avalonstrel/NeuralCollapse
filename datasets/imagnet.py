import os
import torch.distributed as dist
import pickle
import numpy as np
from PIL import Image
from .base_dataset import BaseDataset
from utils.parallel import get_dist_info
from .utils import download_and_extract_archive, check_integrity

class ImageNet(BaseDataset):
    """Since there is prepared dataset class in pytorch, we just wrap it here.
    """
    
    def load_annotations(self):
        rank, world_size = get_dist_info()
        data_infos = []
        class_idx = 0
        data_count = {}
        max_num = 100
        if hasattr(self, 'max_class_num'):
            max_num = self.max_class_num
        for class_label in os.listdir(self.root):
            data_count[class_idx] = 1
            if class_label.startswith('n'):
                class_dir = os.path.join(self.root, class_label)
                file_names = [file_name for file_name in os.listdir(class_dir) if file_name.endswith('JPEG')]
                for file_name in file_names:
                    data_count[class_idx] += 1
                    data_infos.append({'img':os.path.join(class_dir, file_name), 'gt_label':class_idx})
                    if data_count[class_idx] > max_num:
                        break
            class_idx += 1
        print(f'Total Class Num:{class_idx}, max_num:{max_num}')
        self.data_infos = data_infos
        return data_infos

    def __getitem__(self, idx):
        img, label = self.data_infos[idx]['img'], self.data_infos[idx]['gt_label']

        if self.transforms is not None:
            img = self.transforms(Image.open(img).convert('RGB'))
        if self.target_transforms is not None:
            label = self.target_transforms(label)
        return img, label
