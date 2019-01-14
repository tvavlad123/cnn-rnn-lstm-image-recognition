import torch
from torchvision import datasets, transforms


class DataLoader:
    @staticmethod
    def get_train_data(batch_size):
        return torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data',
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batch_size,
            shuffle=True)

    @staticmethod
    def get_test_data(test_batch_size):
        return torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data',
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=test_batch_size,
            shuffle=True)
