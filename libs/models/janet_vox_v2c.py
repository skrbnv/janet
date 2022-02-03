import torch
import torch.nn as nn
''' V8 WA block after each 2d convolution, either funneled or not '''


class Conv2d(nn.Module):
    def __init__(self,
                 input_shape,
                 planes_in,
                 planes_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True,
                 residual=False,
                 extras=None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(planes_in,
                              planes_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(planes_out)
        self.relu = nn.ReLU()
        self.extras = nn.Sequential(*extras) if extras is not None else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.extras is not None:
            x = self.extras(x)
        #x = self.wa(x)
        return x


class Extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adjust = Conv2d(
            input_shape=(64, 192),
            planes_in=1,
            planes_out=128,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
            residual=False,
            extras=[nn.AvgPool2d(3, stride=(1, 3), padding=1)])

        # had to use 128 instead of 64 to prevent division by 0
        seq = [
            Conv2d(input_shape=(int(64 / 2**(i + 1)), int(64 / 2**(i + 1))),
                   planes_in=128 * 2**i,
                   planes_out=128 * 2**(i + 1),
                   kernel_size=5,
                   stride=2,
                   padding=2,
                   bias=False,
                   residual=True,
                   extras=None) for i in range(4)
        ]
        self.funnel = nn.Sequential(*seq)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.adjust(x)
        x = self.funnel(x)
        x = self.avgpool(x)
        return x.flatten(1)


class Classifier(nn.Module):
    def __init__(self, size_in, size_out) -> None:
        super().__init__()
        self.linear1 = nn.Linear(size_in, size_out)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(midsize, size_out)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.dropout(x)
        # x = self.linear2(x)
        return x


class ClassifierWithDropout(nn.Module):
    def __init__(self, size_in, size_inner, dropout, size_out) -> None:
        super().__init__()
        self.linear1 = nn.Linear(size_in, size_inner)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(size_inner, size_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Janet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = Extractor()
        self.classifier = Classifier(size_in=1024, size_out=5994)
        #self.classifier = ClassifierWithDropout(1024, 2048, .5, 5994)

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x
