import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import libs.models.resnet as resnet

ResNetFCOutputSize = 128


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*((x.shape[0], ) + (1, ) + self.shape))


class EmbeddingsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet.resnet34(pretrained=True)
        #self.resnet.fc = nn.Linear(
        #    512 * 1, ResNetFCOutputSize
        #)    # 512 * num_blocks: 1 for resnet34, 4 for resnet50, ? for resnet 101

        ###### converting last d=512 layer to d=128
        self.resnet.fc = nn.Sequential(nn.Linear(512 * 1, 1024),
                                       nn.BatchNorm1d(1024), nn.ReLU(),
                                       nn.Linear(1024, 128))
        self.resnet.conv1 = nn.Conv2d(1,
                                      64,
                                      kernel_size=7,
                                      stride=2,
                                      padding=3,
                                      bias=False)
        ###### Dense layer transforming 80x200 to 224x224
        #self.resnet.conv1 = nn.Sequential(
        #    nn.Flatten(), nn.Linear(int(MEL_BANKS * SLICE_MS / 10), 512),
        #    nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 224 * 224),
        #    View((224, 224)),
        #    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))
        #self.resnet.conv1 = nn.Sequential(
        #    nn.Flatten(), nn.Linear(int(MEL_BANKS * SLICE_MS / 10), 224 * 224, bias = False),
        #    View((224, 224)),
        #    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))

    def forward(self, x):
        x = self.resnet(x)
        return x


class TripletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = EmbeddingsModel()

        #self.bn1 = nn.BatchNorm1d(num_features=128)
        #self.fc1 = nn.Linear(ResNetFCOutputSize, 128)

    def forward(self, x, y, z):
        x = self.innerModel(x)
        y = self.innerModel(y)
        z = self.innerModel(z)
        return x, y, z

    def innerModel(self, x):
        x = self.basemodel(x)
        #x = x / torch.linalg.norm(x)
        return x
