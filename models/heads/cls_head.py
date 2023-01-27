import torch
import torch.nn as nn

class ClsHead(nn.Module):
    def __init__(self, 
            in_channels,
            num_classes, # num_classes
        ):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

        self.init_weights()

    def init_weights(self):
        def km_init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
        self.fc.apply(km_init_weights)

    def forward(self, x):
        return self.fc(x)
