
import torch


class BaseModel(torch.nn.Module):
    """
    A base model to be implemeted for different tasks.
    """
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    





