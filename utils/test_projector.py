# utils/test_projector.py

import torch
from models.projector import Projector

proj = Projector()

x = torch.randn(4, 512)
out = proj(x)

print("Projector output shape:", out.shape)
print("Expected shape: (4, 128)")