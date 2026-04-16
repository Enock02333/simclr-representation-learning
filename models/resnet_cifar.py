# models/resnet_cifar.py

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 modified for CIFAR-10:
    - 3x3 conv, stride=1
    - no max pooling
    - output feature dim = 512
    """

    def __init__(self):
        super().__init__()

        # Create ResNet-18 WITHOUT pretrained weights
        self.encoder = resnet18(weights=None)

        # Modify the stem
        self.encoder.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.encoder.maxpool = nn.Identity()

        # Remove classification head
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        """
        Output: (batch_size, 512)
        """
        return self.encoder(x)