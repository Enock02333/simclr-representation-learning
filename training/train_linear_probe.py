# training/train_linear_probe.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.resnet_cifar import ResNet18CIFAR
from datasets.eval_dataset import get_cifar10_eval


CKPT_PATH = "outputs/simclr_baseline/checkpoints/epoch_200.pth"


def train_linear_probe(
    epochs=100,
    batch_size=256,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Using device: {device}")

    # -----------------------
    # Datasets
    # -----------------------
    train_dataset = get_cifar10_eval("data", train=True)
    test_dataset = get_cifar10_eval("data", train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # -----------------------
    # Backbone (FROZEN)
    # -----------------------
    backbone = ResNet18CIFAR().to(device)
    backbone.eval()

    ckpt = torch.load(CKPT_PATH, map_location=device)
    backbone.load_state_dict(ckpt["backbone"])

    for param in backbone.parameters():
        param.requires_grad = False

    # -----------------------
    # Linear classifier
    # -----------------------
    classifier = nn.Linear(512, 10).to(device)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=lr,
        weight_decay=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                feats = backbone(x)

            logits = classifier(feats)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -----------------------
        # Evaluation
        # -----------------------
        classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                feats = backbone(x)
                logits = classifier(feats)

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = 100.0 * correct / total
        print(f"Epoch {epoch:03d}: Loss={avg_loss:.4f}, Test Acc={acc:.2f}%")

    print("✅ Linear probing finished.")


if __name__ == "__main__":
    train_linear_probe()
