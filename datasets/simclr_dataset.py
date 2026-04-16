# datasets/simclr_dataset.py

from torchvision.datasets import CIFAR10

class SimCLRCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__(
            root=root,
            train=train,
            transform=None,  # IMPORTANT: we override transform behavior
            download=download
        )
        self.simclr_transform = transform

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        view1, view2 = self.simclr_transform(image)
        return view1, view2
