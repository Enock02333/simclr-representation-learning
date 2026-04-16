# datasets/eval_dataset.py

from torchvision.datasets import CIFAR10
from torchvision import transforms

def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

def get_cifar10_eval(root, train):
    return CIFAR10(
        root=root,
        train=train,
        transform=get_eval_transform(),
        download=True
    )