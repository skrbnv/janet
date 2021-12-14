from torch import Tensor, vstack
import torch.nn as nn
import torch.nn.functional as F

# bottlenecks > equal?


class Bottleneck(nn.Module):
    def __init__(self, inplanes=128, width=64, bias=False):
        super().__init__()
        self.conv1x1A = nn.Conv2d(inplanes,
                                  width,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=bias)
        self.bnA = nn.BatchNorm2d(width)
        self.conv3x3 = nn.Conv2d(width,
                                 width,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=bias)
        self.bnM = nn.BatchNorm2d(width)
        self.conv1x1Z = nn.Conv2d(width,
                                  inplanes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=bias)
        self.bnZ = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1x1A(x)
        out = self.bnA(out)
        out = self.relu(out)
        out = self.conv3x3(out)
        out = self.bnM(out)
        out = self.relu(out)
        out = self.conv1x1Z(out)
        out = self.bnZ(out)
        out += identity
        out = self.relu(out)
        return out


class EmbeddingsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # (64,1,64,192)

        self.conv1 = nn.Conv2d(1,
                               64,
                               kernel_size=5,
                               stride=(1, 3),
                               padding=2,
                               bias=False)
        # (64,64,64,64)
        self.bn64 = nn.BatchNorm2d(64)

        self.avgpool1 = nn.AvgPool2d(2)
        # (64,64,32,32)

        self.conv2 = nn.Conv2d(64,
                               128,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        # (64,128,16,16)
        self.bn128 = nn.BatchNorm2d(128)

        self.block128 = Bottleneck(inplanes=128, width=64, bias=False)
        # (64,128,16,16)

        self.conv3 = nn.Conv2d(128,
                               256,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        # (64,256,8,8)
        self.bn256 = nn.BatchNorm2d(256)

        self.block256 = Bottleneck(inplanes=256, width=64, bias=False)
        # (64,256,8,8)

        self.conv4 = nn.Conv2d(256,
                               512,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        # (64,512,4,4)
        self.bn512 = nn.BatchNorm2d(512)

        self.block512 = Bottleneck(inplanes=512, width=64, bias=False)
        # (64,512,4,4)

        self.conv5 = nn.Conv2d(512,
                               1024,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        # (64,1024,2,2)
        self.bn1024 = nn.BatchNorm2d(1024)

        self.avgpool2 = nn.AvgPool2d(2)
        # (64,1024,1,1)

        self.fc1 = nn.Linear(1024, 630)
        #self.bn1024f = nn.BatchNorm1d(512)
        ##self.dfc1 = nn.Dropout(p=0.1)
        #self.fc2 = nn.Linear(512, 630)

    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn64(x)
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn128(x)
        x = F.relu(x)
        #for i in range(2):
        #x = self.block128(x)
        x = self.conv3(x)
        x = self.bn256(x)
        x = F.relu(x)
        #for i in range(2):
        #x = self.block256(x)
        x = self.conv4(x)
        x = self.bn512(x)
        x = F.relu(x)
        #for i in range(2):
        #x = self.block512(x)
        x = self.conv5(x)
        x = self.bn1024(x)
        x = F.relu(x)
        # (64,1024,2,3)
        x = self.avgpool2(x)
        # (64,1024,1,1)
        x = self.flatten(x)
        # (64,1024)
        x = self.fc1(x)
        # (64,512)
        #x = self.bn1024f(x)
        #x = self.dfc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)
        # (64,128)
        #x = F.normalize(x)
        return x

    def flatten(self, x):
        return x.view(x.shape[0], -1)


class TripletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = EmbeddingsModel()

    def forward(self, x, y, z):
        anchor = self.innerModel(x)
        positive = self.innerModel(y)
        negative = self.innerModel(z)
        return anchor, positive, negative

    def innerModel(self, x):
        return self.basemodel(x)
