# utils/test_nt_xent.py

import torch
from losses.nt_xent import NTXentLoss

loss_fn = NTXentLoss(temperature=0.5)

batch_size = 4
z1 = torch.randn(batch_size, 128)
z2 = torch.randn(batch_size, 128)

loss = loss_fn(z1, z2)

print("NT-Xent loss:", loss.item())