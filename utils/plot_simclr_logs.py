# utils/plot_simclr_logs.py

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "outputs/simclr_baseline"
LOG_DIR = os.path.join(BASE_DIR, "logs")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)


def plot_loss():
    loss_csv = os.path.join(LOG_DIR, "loss.csv")
    df = pd.read_csv(loss_csv)

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["loss"], label="SimCLR Loss")
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.title("SimCLR Training Loss")
    plt.grid(True)
    plt.legend()

    path = os.path.join(PLOT_DIR, "loss_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {path}")


def plot_knn():
    knn_csv = os.path.join(LOG_DIR, "knn.csv")
    df = pd.read_csv(knn_csv)

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["knn_acc"], marker="o", label="kNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("kNN Monitor Accuracy")
    plt.grid(True)
    plt.legend()

    path = os.path.join(PLOT_DIR, "knn_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {path}")


if __name__ == "__main__":
    plot_loss()
    plot_knn()