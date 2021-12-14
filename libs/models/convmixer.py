from numpy import inner
import torch.nn as nn
from timm import create_model as timm_create_model


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim,
              depth,
              input_channels=3,
              kernel_size=9,
              patch_size=7,
              n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(input_channels,
                  dim,
                  kernel_size=patch_size,
                  stride=patch_size), nn.GELU(), nn.BatchNorm2d(dim), *[
                      nn.Sequential(
                          Residual(
                              nn.Sequential(
                                  nn.Conv2d(dim,
                                            dim,
                                            kernel_size,
                                            groups=dim,
                                            padding="same"), nn.GELU(),
                                  nn.BatchNorm2d(dim))),
                          nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(),
                          nn.BatchNorm2d(dim)) for i in range(depth)
                  ], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.Linear(dim, n_classes))


class Model(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 input_channels=3,
                 kernel_size=9,
                 patch_size=7,
                 n_classes=1000):
        super().__init__()
        self.basemodel = ConvMixer(dim, depth, input_channels, kernel_size,
                                   patch_size, n_classes)

    def forward(self, x):
        return self.innerModel(x)

    def innerModel(self, x):
        return self.basemodel(x)


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = timm_create_model('convmixer_768_32', pretrained=True)

    def forward(self, x):
        return self.innerModel(x)

    def innerModel(self, x):
        return self.basemodel(
            x.broadcast_to(x.shape[0], 3, x.shape[2], x.shape[3]))
