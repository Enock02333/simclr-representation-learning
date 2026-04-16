# utils/knn_monitor.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter


@torch.no_grad()
def knn_monitor(
    backbone,
    train_dataset,
    test_dataset,
    k=20,
    batch_size=256,
    device="cuda"
):
    """
    kNN monitor for SimCLR.
    Uses backbone features (NOT projector).
    """

    backbone.eval()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # --- Extract train features ---
    train_features = []
    train_labels = []

    for x, y in train_loader:
        x = x.to(device)
        feats = backbone(x)
        feats = F.normalize(feats, dim=1)

        train_features.append(feats.cpu())
        train_labels.append(y)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # --- kNN classification ---
    correct = 0
    total = 0

    for x, y in test_loader:
        x = x.to(device)
        feats = backbone(x)
        feats = F.normalize(feats, dim=1)

        # cosine similarity
        sim = torch.mm(feats.cpu(), train_features.T)
        _, indices = sim.topk(k=k, dim=1)

        neighbors = train_labels[indices]

        for i in range(neighbors.size(0)):
            pred = Counter(neighbors[i].tolist()).most_common(1)[0][0]
            correct += int(pred == y[i].item())
            total += 1

    acc = 100.0 * correct / total
    return acc