# utils/test_dataset.py

import torch
from datasets.simclr_dataset import SimCLRCIFAR10
from datasets.simclr_transforms import SimCLRTransform
from torch.utils.data import DataLoader

dataset = SimCLRCIFAR10(
    root="data",
    train=True,
    transform=SimCLRTransform(),
    download=True
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

view1, view2 = next(iter(loader))

print("View1 shape:", view1.shape)
print("View2 shape:", view2.shape)
print("Are views identical?", torch.allclose(view1, view2))