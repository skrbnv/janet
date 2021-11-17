import torch.nn as nn
from resnet import resnet18, _resnet, BasicBlock


class ResNet18_64x192_kernel5_nomaxpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = resnet18(pretrained=False)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False), )
        self.basemodel.maxpool = nn.ReLU()
        self.basemodel.fc = nn.Linear(512, 128)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet10_64x192_kernel5_nomaxpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet10',
                                 BasicBlock, [1, 1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False), )
        self.basemodel.maxpool = nn.ReLU()
        self.basemodel.fc = nn.Linear(512, 128)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))
