import numpy as np
import torchvision


class DataLoader:
    def __init__(self):
        self.train_data = np.zeros(1)

    def get_mnist(self):
        self.train_data = torchvision.datasets.MNIST(
            root='../input/mnist/mnist/',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

    def get_train_data(self):
        return self.train_data
