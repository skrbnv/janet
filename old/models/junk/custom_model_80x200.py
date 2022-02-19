import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as D
import random
import numpy as np


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,
                               32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 1)
        self.bn3 = nn.BatchNorm2d(128)

        # (64,128,20,50)
        self.block2050 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (64,128,20,50)
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (64,128,20,50)
        )

        self.conv_ab1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn_ab1 = nn.BatchNorm2d(256)
        # (64,256,10,25)

        self.block1025 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # (64,256,10,25)
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # (64,256,10,25)
        )

        self.conv_ab2 = nn.Conv2d(256,
                                  512,
                                  kernel_size=3,
                                  stride=2,
                                  padding=(1, 0))
        self.bn_ab2 = nn.BatchNorm2d(512)
        # (64,512,5,12)

        self.block512 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # (64,256,3,6)
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # (64,256,3,6)
        )
        self.conv_ab3 = nn.Conv2d(512,
                                  1024,
                                  kernel_size=3,
                                  stride=2,
                                  padding=(1, 0))
        self.bn_ab3 = nn.BatchNorm2d(1024)

        self.conv_f1 = nn.Conv2d(1024,
                                 2048,
                                 kernel_size=3,
                                 stride=2,
                                 padding=(1, 0))
        self.bn2048 = nn.BatchNorm2d(2048)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.2)
        self.dense1 = nn.Linear(2048, 1024)
        self.dense2 = nn.Linear(1024, 128)

    def forward(self, x):
        # (64,1,80,200)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        # (64,32,40,100)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # (64,64,20,50)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # (64,128,20,50)
        for i in range(3):
            identity = x
            x = self.block2050(x)
            x += identity
        # (64,128,20,50)
        x = self.conv_ab1(x)
        x = self.bn_ab1(x)
        x = self.relu(x)
        # (64,256,10,25)
        for i in range(6):
            identity = x
            x = self.block1025(x)
            x += identity
        x = self.conv_ab2(x)
        x = self.bn_ab2(x)
        x = self.relu(x)
        #(64, 512, 5, 12)
        for i in range(3):
            identity = x
            x = self.block512(x)
            x += identity
        #(64, 512, 5, 12)
        x = self.conv_ab3(x)
        x = self.bn_ab3(x)
        x = self.relu(x)
        #(64, 1024, 3, 5)
        x = self.conv_f1(x)
        x = self.bn2048(x)
        x = self.relu(x)
        #(64, 2048, 2, 2)
        x = self.maxpool(x)
        x = self.flatten(x)
        #(64, 2048)
        x = self.dense1(x)
        #(64, 512)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        #(64, 128)
        x = F.normalize(x)
        return x

    def flatten(self, x):
        #print(x.shape)
        return x.view(x.shape[0], -1)
        #return np.reshape(ndim_array, (ndim_array.shape[0],ndim_array.shape[1]*ndim_array.shape[2]))


class TripletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.innerModel = BaseModel()

    def forward(self, x, y, z):
        x, y, z = self.innerModel(x), self.innerModel(y), self.innerModel(z)
        return x, y, z