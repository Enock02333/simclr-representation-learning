# training/train_supervised.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.resnet_cifar import ResNet18CIFAR
from datasets.eval_dataset import get_cifar10_eval


def train_supervised(
    epochs=200,
    batch_size=128,
    lr=3e-4,
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
    # Model
    # -----------------------
    backbone = ResNet18CIFAR().to(device)
    classifier = nn.Linear(512, 10).to(device)
    model = nn.Sequential(backbone, classifier)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        # -----------------------
        # Evaluation
        # -----------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                preds = logits.argmax(dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)

        print(
            f"Epoch {epoch:03d}: "
            f"Train Loss={avg_loss:.4f}, "
            f"Test Acc={acc:.2f}%, "
            f"Best={best_acc:.2f}%"
        )

    print(f"✅ Supervised training finished. Best Test Acc = {best_acc:.2f}%")


if __name__ == "__main__":
    train_supervised()