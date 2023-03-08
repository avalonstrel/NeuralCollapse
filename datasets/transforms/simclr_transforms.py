import numpy as np
import torchvision.transforms as transforms

np.random.seed(0)

def simclr_transform(size, s=1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return transforms.Compose([transforms.RandomResizedCrop(size=size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([color_jitter], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.GaussianBlur(kernel_size=int(0.1 * size[0])),
                                transforms.ToTensor()])

class SimCLRTransforms(object):
    """
    SimCLR transforms.
    """
    def __init__(self, size, n_views=2) -> None:
        self.base_transform = simclr_transform(size=size)
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
        
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
