# losses/nt_xent.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: (batch_size, dim)
        """
        batch_size = z1.size(0)

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate
        z = torch.cat([z1, z2], dim=0)  # (2N, dim)

        # Similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # (2N, 2N)

        # Mask self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs
        positives = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ], dim=0)

        # Denominator
        loss = -positives + torch.logsumexp(sim, dim=1)

        return loss.mean()