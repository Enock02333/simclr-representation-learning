# utils/test_backbone.py

import torch
from models.resnet_cifar import ResNet18CIFAR

model = ResNet18CIFAR()

x = torch.randn(4, 3, 32, 32)
out = model(x)

print("Output shape:", out.shape)
print("Expected shape: (4, 512)")