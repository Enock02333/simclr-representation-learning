# training/train_simclr.py

import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.simclr_dataset import SimCLRCIFAR10
from datasets.simclr_transforms import SimCLRTransform
from datasets.eval_dataset import get_cifar10_eval

from models.resnet_cifar import ResNet18CIFAR
from models.projector import Projector
from losses.nt_xent import NTXentLoss
from utils.knn_monitor import knn_monitor


# -----------------------
# Paths
# -----------------------
BASE_DIR = "outputs/simclr_baseline"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def save_checkpoint(epoch, backbone, projector, optimizer):
    path = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth")
    torch.save({
        "epoch": epoch,
        "backbone": backbone.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict()
    }, path)
    print(f"[Checkpoint] Saved: {path}")


def load_checkpoint(path, backbone, projector, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    backbone.load_state_dict(ckpt["backbone"])
    projector.load_state_dict(ckpt["projector"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Checkpoint] Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]


def train_simclr(
    epochs=200,
    batch_size=128,
    lr=3e-4,
    temperature=0.5,
    knn_k=20,
    knn_interval=5,
    resume_ckpt=None,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Using device: {device}")

    # -----------------------
    # Datasets
    # -----------------------
    simclr_dataset = SimCLRCIFAR10(
        root="data",
        train=True,
        transform=SimCLRTransform(),
        download=True
    )

    simclr_loader = DataLoader(
        simclr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    train_eval = get_cifar10_eval("data", train=True)
    test_eval = get_cifar10_eval("data", train=False)

    # -----------------------
    # Models
    # -----------------------
    backbone = ResNet18CIFAR().to(device)
    projector = Projector().to(device)

    criterion = NTXentLoss(temperature=temperature)

    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=lr,
        weight_decay=1e-6
    )

    start_epoch = 1

    # -----------------------
    # Resume if needed
    # -----------------------
    if resume_ckpt is not None:
        start_epoch = load_checkpoint(
            resume_ckpt, backbone, projector, optimizer
        ) + 1

    # -----------------------
    # CSV loggers
    # -----------------------
    loss_log_path = os.path.join(LOG_DIR, "loss.csv")
    knn_log_path = os.path.join(LOG_DIR, "knn.csv")

    if start_epoch == 1:
        with open(loss_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss"])
        with open(knn_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "knn_acc"])

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(start_epoch, epochs + 1):
        backbone.train()
        projector.train()

        total_loss = 0.0
        pbar = tqdm(simclr_loader, desc=f"Epoch [{epoch}/{epochs}]")

        for view1, view2 in pbar:
            view1 = view1.to(device)
            view2 = view2.to(device)

            h1 = backbone(view1)
            h2 = backbone(view2)

            z1 = projector(h1)
            z2 = projector(h2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(simclr_loader)
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

        # log loss
        with open(loss_log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_loss])

        # -----------------------
        # kNN monitor
        # -----------------------
        if epoch % knn_interval == 0:
            acc = knn_monitor(
                backbone,
                train_eval,
                test_eval,
                k=knn_k,
                device=device
            )
            print(f"kNN accuracy @ epoch {epoch}: {acc:.2f}%")

            with open(knn_log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, acc])

        # -----------------------
        # Save checkpoints
        # -----------------------
        if epoch in {50, 100, 200}:
            save_checkpoint(epoch, backbone, projector, optimizer)

    print("SimCLR training finished.")


if __name__ == "__main__":
    train_simclr()