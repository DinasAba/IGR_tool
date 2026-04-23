import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    # 4 konv + 3 FC sluoksniai. Naudojamas MNIST/Fashion-MNIST/OCTMNIST/PneumoniaMNIST (1x28x28).

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14x14
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)  # 7x7

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CIFAR10ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18

        self.net = resnet18(num_classes=10)
        # 32x32 vaizdams 7x7 kernelio per daug
        self.net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)


class DermaMNISTResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18

        self.net = resnet18(num_classes=7)
        self.net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)


class RetinaMNISTResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18

        self.net = resnet18(num_classes=5)
        self.net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)


def build_model(name: str):
    if name == "mnist_cnn":
        return MNISTCNN(num_classes=10)
    if name == "octmnist_cnn":
        return MNISTCNN(num_classes=4)
    if name == "pneumoniamnist_cnn":
        return MNISTCNN(num_classes=2)
    if name == "cifar10_resnet18":
        return CIFAR10ResNet18()
    if name == "dermamnist_resnet18":
        return DermaMNISTResNet18()
    if name == "retinamnist_resnet18":
        return RetinaMNISTResNet18()
    raise ValueError(f"Unknown model: {name}")
