from torch import Tensor, vstack
import torch.nn as nn
import torch.nn.functional as F

# bottlenecks > equal?


def conv1x1(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes,
                     outplanes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def conv3x3(inplanes, outplanes, stride=2, kernel=3, padding=1):
    return nn.Conv2d(inplanes,
                     outplanes,
                     kernel_size=kernel,
                     stride=stride,
                     padding=padding,
                     bias=False)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, down, custom_kernel, custom_padding):
        super().__init__()
        width = int(inplanes / 2) if down else int(inplanes / 4)
        if custom_padding is False:
            custom_padding = 1
        if custom_kernel is False:
            custom_kernel = 3
        self.downsample = True if down else False
        self.downsample_identity = nn.Sequential(
            nn.Conv2d(inplanes,
                      width * 4,
                      stride=2,
                      kernel_size=custom_kernel,
                      padding=custom_padding), nn.BatchNorm2d(width * 4))
        self.conv1x1A = conv1x1(inplanes, width)
        self.bnA = nn.BatchNorm2d(width)
        if down:
            self.conv3x3 = conv3x3(width,
                                   width,
                                   stride=2,
                                   kernel=custom_kernel,
                                   padding=custom_padding)
        else:
            self.conv3x3 = conv3x3(width, width, stride=1)

        self.bnM = nn.BatchNorm2d(width)
        self.conv1x1Z = conv1x1(width, width * 4)
        self.bnZ = nn.BatchNorm2d(width * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.downsample:
            identity = self.downsample_identity(x)
        else:
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


def make_block(inplanes, repeats, custom_kernel=False, custom_padding=False):

    layers = [
        Bottleneck(inplanes,
                   down=True,
                   custom_kernel=custom_kernel,
                   custom_padding=custom_padding)
    ]
    for _ in range(1, repeats):
        layers.append(
            Bottleneck(inplanes * 2,
                       down=False,
                       custom_kernel=custom_kernel,
                       custom_padding=custom_padding))
    return nn.Sequential(*layers)


class EmbeddingsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # (64,1,80,200)

        self.conv1 = nn.Conv2d(1,
                               64,
                               kernel_size=(3, 5),
                               stride=(1, 2),
                               padding=(1, 2),
                               bias=False)
        # (64,64,80,100)
        self.bn64 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,
                               128,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        # (64,128,40,50)
        self.bn128 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128,
                               256,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        # (64,256,20,25)
        self.bn256 = nn.BatchNorm2d(256)

        self.block256 = make_block(256,
                                   3,
                                   custom_kernel=3,
                                   custom_padding=(1, 0))
        # (64,512,10,12)
        self.bn512 = nn.BatchNorm2d(512)
        self.block512 = make_block(512, 4)
        # (64,1024,5,6)
        self.bn1024 = nn.BatchNorm2d(1024)
        self.block1024 = make_block(1024, 3, custom_padding=(1, 2))
        # (64,2048,2,3)
        self.bn2048 = nn.BatchNorm2d(2048)
        self.avgpool310 = nn.AvgPool2d(3, padding=(1, 0))
        # (64,1024,1,1)
        self.fc1 = nn.Linear(2048, 1024, bias=False)
        self.bn1024f = nn.BatchNorm1d(1024)
        #self.dfc1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn64(x)
        x = F.relu(x)
        # (64,32,80,100)
        x = self.conv2(x)
        x = self.bn128(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn256(x)
        x = F.relu(x)
        # (64,128,20,25)
        x = self.block256(x)
        x = self.bn512(x)
        # (64,512,10,12)
        x = self.block512(x)
        x = self.bn1024(x)
        # (64,1024,5,6)
        x = self.block1024(x)
        x = self.bn2048(x)
        # (64,1024,2,3)
        x = self.avgpool310(x)
        # (64,1024,1,1)
        x = self.flatten(x)
        # (64,1024)
        x = self.fc1(x)
        # (64,512)
        x = self.bn1024f(x)
        #x = self.dfc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # (64,128)
        x = F.normalize(x)
        return x

    def flatten(self, x):
        return x.view(x.shape[0], -1)


class DualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = EmbeddingsModel()

    def forward(self, p, n):
        p = self.innerModel(p)
        n = self.innerModel(n)
        return p, n

    def innerModel(self, x):
        return self.basemodel(x)
