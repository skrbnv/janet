import torch
import torch.nn as nn
from libs.semiatt import DirectionalConv2d


class Extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adjust = DirectionalConv2d(
            midshape=(64, 64),
            planes_in=1,
            planes_out=128,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
            extras=[nn.AvgPool2d(3, stride=(1, 3), padding=1)])

        seq = [
            DirectionalConv2d(midshape=(int(64 / 4**(i + 1)),
                                        int(64 / 4**(i + 1))),
                              planes_in=128 * 2**i,
                              planes_out=128 * 2**(i + 1),
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False,
                              extras=[nn.AvgPool2d(2, 2)]) for i in range(3)
        ]
        seq[-1].wm = nn.Identity()
        self.funnel = nn.Sequential(*seq)

    def forward(self, x):
        x = self.adjust(x)
        x = self.funnel(x)
        return x.flatten(1)


class Classifier(nn.Module):
    def __init__(self, size_in, size_out) -> None:
        super().__init__()
        self.linear = nn.Linear(size_in, size_out)

    def forward(self, x):
        x = self.linear(x)
        return x


class Janet(nn.Module):
    def __init__(self, midsize=1024, num_classes=5994) -> None:
        super().__init__()
        self.extractor = Extractor()
        self.classifier = Classifier(size_in=midsize, size_out=num_classes)

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x
