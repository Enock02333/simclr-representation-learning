# Self-Supervised Representation Learning with SimCLR

This repository provides a clean and reproducible implementation of **SimCLR**, a self-supervised contrastive learning framework for visual representation learning. The project focuses on learning transferable image representations without labels and evaluating them using kNN monitoring, linear probing, and comparison with supervised training.

The implementation is designed to reflect research-level experimental practice rather than a coursework-only submission.

---

## Motivation

In many computer vision applications such as **pattern recognition** and **image segmentation**, acquiring large-scale labeled datasets is expensive and time-consuming. Self-supervised learning addresses this challenge by learning meaningful representations directly from unlabeled data.

This project investigates whether contrastive self-supervised learning can produce representations that are competitive with, or superior to, those learned via supervised training from scratch.

---

## Method Overview

- **Backbone:** Modified ResNet-18 (CIFAR-10 compatible)
- **Self-Supervised Method:** SimCLR
- **Loss Function:** NT-Xent
- **Dataset:** CIFAR-10
- **Evaluation:**
  - k-Nearest-Neighbor (kNN) monitoring
  - Linear probing on frozen representations
  - Fully supervised baseline comparison

All models are trained **from scratch**, without pretrained weights.

---

## Experimental Results

| Model | Test Accuracy (%) |
|------|-------------------|
| Random Initialization + Linear | ~ | **86.93** || Random Initialization + Linear | ~10 |
| **Supervised Training (from scratch)** | **84.17** |

Self-supervised contrastive learning produces more transferable representations than supervised training under identical conditions.

---

## Training Dynamics

### SimCLR Training Loss
![SimCLR Loss](outputs/simclr_baseline/plots/loss_curve.png)

### kNN Monitor Accuracy
![kNN Accuracy](outputs/simclr_baseline/plots/knn_curve.png)

---

## Repository Structure


datasets/     # Data loaders and augmentations
models/       # Backbone and projection head
losses/       # Contrastive loss
training/     # Training scripts
utils/        # Evaluation and visualization utilities
outputs/      # Logs and plots
report/       # Final report

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt

Train SimCLR
Shellpython training/train_simclr.pyShow more lines
Linear Probing
Shellpython training/train_linear_probe.pyShow more lines
Supervised Baseline
Shellpython training/train_supervised.pyShow more lines

References

Chen et al., A Simple Framework for Contrastive Learning of Visual Representations, ICML 2020.
Krizhevsky & Hinton, Learning Multiple Layers of Features from Tiny Images, 2009.
He et al., Deep Residual Learning for Image Recognition, CVPR 2016.
Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, NeurIPS 2019.


---

## 6️⃣ Git Initialization & First Commit (Terminal)

From inside `simclr-representation-learning/`:

```bash
git init
git add .
git commit -m "Initial SimCLR implementation with evaluation and analysis"