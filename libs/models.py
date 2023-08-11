import torch.nn as nn
from libs.alignment import DirectionalConv2d, PlanarConv2d


class ExtractorSimple(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adjust = nn.Sequential(nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=(1, 3), padding=1))

        seq = [
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
        ]
        seq[-1].alignment = nn.Identity()
        self.funnel = nn.Sequential(*seq)

    def forward(self, x):
        x = self.adjust(x)
        x = self.funnel(x)
        return x.flatten(1)


class ExtractorDirectional(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adjust = DirectionalConv2d(
            shape=(64, 64),
            in_channels=1,
            out_channels=128,
            kernel=7,
            stride=1,
            padding=3,
            extras=[nn.AvgPool2d(3, stride=(1, 3), padding=1)])

        seq = [
            DirectionalConv2d(shape=(int(64 / 4**(i + 1)), int(64 / 4**(i + 1))),
                              in_channels=128 * 2**i,
                              out_channels=128 * 2**(i + 1),
                              kernel=3,
                              stride=2,
                              padding=1,
                              extras=[nn.AvgPool2d(2, 2)]) for i in range(3)
        ]
        seq[-1].alignment = nn.Identity()
        self.funnel = nn.Sequential(*seq)

    def forward(self, x):
        x = self.adjust(x)
        x = self.funnel(x)
        return x.flatten(1)


class ExtractorPlanar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adjust = PlanarConv2d(
            shape=(64, 64),
            in_channels=1,
            out_channels=128,
            kernel=7,
            stride=1,
            padding=3,
            extras=[nn.AvgPool2d(3, stride=(1, 3), padding=1)])

        seq = [
            PlanarConv2d(shape=(int(64 / 4**(i + 1)), int(64 / 4**(i + 1))),
                         in_channels=128 * 2**i,
                         out_channels=128 * 2**(i + 1),
                         kernel=3,
                         stride=2,
                         padding=1,
                         extras=[nn.AvgPool2d(2, 2)]) for i in range(3)
        ]
        seq[-1].alignment = nn.Identity()
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
        self.extractor = ExtractorDirectional()
        self.classifier = Classifier(size_in=midsize, size_out=num_classes)

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x
