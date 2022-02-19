#from torch import Tensor, vstack
import torch.nn as nn
import torch.nn.functional as F


class Extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,
                               64,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=False)
        self.bn64 = nn.BatchNorm2d(64)
        self.avgpool1 = nn.AvgPool2d(3, stride=(1, 3), padding=1)
        self.conv2 = nn.Conv2d(64,
                               128,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        self.bn128 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,
                               256,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        self.bn256 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,
                               512,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        self.bn512 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512,
                               1024,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        self.bn1024 = nn.BatchNorm2d(1024)
        self.avgpool2 = nn.AvgPool2d(4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn64(x)
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn128(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn256(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn512(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn1024(x)
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        return x


class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.dfc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2048, 5994)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dfc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Model_v7_ex_v6f(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = Extractor()
        self.classifier = Classifier()

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x
