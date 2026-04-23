import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def _make_loaders(train_ds, test_ds, batch_size, num_workers):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def get_mnist_loaders(batch_size: int, num_workers: int = 2):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    return _make_loaders(train_ds, test_ds, batch_size, num_workers)


def get_fashion_mnist_loaders(batch_size: int, num_workers: int = 2):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=tfm
    )
    test_ds = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=tfm
    )
    return _make_loaders(train_ds, test_ds, batch_size, num_workers)


def get_cifar10_loaders(batch_size: int, num_workers: int = 2):
    train_tfm = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_tfm
    )
    test_ds = datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_tfm
    )
    return _make_loaders(train_ds, test_ds, batch_size, num_workers)


class _MedMNISTWrapper:
    # MedMNIST graz int label'us (kaip torchvision)
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, int(label.squeeze())


def _medmnist_loaders(MNISTClass, n_classes, train_tfm, test_tfm, batch_size):
    # num_workers=0, nes Windows multiprocessing sukelia hang'us
    raw_train = MNISTClass(split="train", transform=train_tfm, download=True, root="data", size=28)
    train_ds = _MedMNISTWrapper(raw_train)
    test_ds = _MedMNISTWrapper(
        MNISTClass(split="test", transform=test_tfm, download=True, root="data", size=28)
    )

    # weighted sampler disbalanso atveju
    labels = raw_train.labels.squeeze()
    class_counts = np.bincount(labels, minlength=n_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, test_loader


def get_dermamnist_loaders(batch_size: int, num_workers: int = 2):
    # DermaMNIST: 7 klases, 3x28x28
    from medmnist import DermaMNIST
    train_tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])
    test_tfm = transforms.Compose([transforms.ToTensor()])
    return _medmnist_loaders(DermaMNIST, 7, train_tfm, test_tfm, batch_size)


def get_octmnist_loaders(batch_size: int, num_workers: int = 2):
    # OCTMNIST: 4 klases, 1x28x28, ~109k pavyzdziu.
    # Weighted sampling nenaudojam - didelis rinkinys, disbalansas nedidelis (~47% max klase).
    from medmnist import OCTMNIST
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = _MedMNISTWrapper(
        OCTMNIST(split="train", transform=tfm, download=True, root="data", size=28)
    )
    test_ds = _MedMNISTWrapper(
        OCTMNIST(split="test", transform=tfm, download=True, root="data", size=28)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, test_loader


def get_pneumoniamnist_loaders(batch_size: int, num_workers: int = 2):
    # PneumoniaMNIST: 2 klases, 1x28x28, ~5.8k pavyzdziu; disbalansuotas
    from medmnist import PneumoniaMNIST
    train_tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    test_tfm = transforms.Compose([transforms.ToTensor()])
    return _medmnist_loaders(PneumoniaMNIST, 2, train_tfm, test_tfm, batch_size)


DATASETS = {
    "mnist": get_mnist_loaders,
    "fashion_mnist": get_fashion_mnist_loaders,
    "cifar10": get_cifar10_loaders,
    "dermamnist": get_dermamnist_loaders,
    "octmnist": get_octmnist_loaders,
    "pneumoniamnist": get_pneumoniamnist_loaders,
}


def get_loaders(name: str, batch_size: int, num_workers: int = 2):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")
    return DATASETS[name](batch_size, num_workers)
