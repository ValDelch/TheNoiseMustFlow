"""
MNIST.py

A simple MNIST dataset example.
"""


import torch
from torchvision import datasets, transforms


class MNIST(torch.utils.data.Dataset):
    """
    MNIST dataset class.
    """

    def __init__(self, root: str = './data', train: bool = True, download: bool = True,
                 d_context: int = 128):
        """
        Initializes the MNIST dataset.

        The dataset is downloaded if not already present.

        Note that the images are resized to 64x64 pixels for convenience with the model
        down sampling operations.

        Args:
            root (str): Root directory where the dataset is stored.
            train (bool): If True, loads the training set; otherwise, loads the test set.
            download (bool): If True, downloads the dataset if not already present.
            d_context (int): Dimension of the context vector. Here, context is the one-hot
                encoded label of the digit (0-9), padded to the specified dimension.
        """
        # Reshape to 32x32 for convenience with the models
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=self.transform
        )
        self.d_context = d_context

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        image, label = self.dataset[idx]
        c = torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()
        c = torch.nn.functional.pad(c, (0, self.d_context - 10), value=0.0)
        return {
            'image': image,
            'context': c[None, :], # Add a seq_len dimension
            'label': label
        }