import torch
import torch.nn as nn


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3,
                                groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim,
            4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.residual = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.residual(x)
        return x


class Extractor(nn.Module):
    def __init__(self, chseq=[1, 64, 128, 256, 512]) -> None:
        super().__init__()
        self.stem = nn.Sequential(*[
            nn.Conv2d(chseq[0], chseq[1], kernel_size=4, stride=4, padding=0),
            nn.LayerNorm([64, 16, 48]),
        ])
        '''
        self.downsample = nn.ModuleList()
        for i in range(1, 4):
            downsample_layer = nn.Sequential(
                nn.LayerNorm(
                    [chseq[i], int(16 / 2**i),
                     int(48 / 2**i)], eps=1e-6),
                nn.Conv2d(chseq[i], chseq[i + 1], kernel_size=2, stride=2),
            )
            self.downsample.append(downsample_layer)
        '''
        self.downsample1 = nn.Sequential(
            nn.LayerNorm([64, 16, 48], eps=1e-6),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
        )
        self.downsample2 = nn.Sequential(
            nn.LayerNorm([128, 8, 24], eps=1e-6),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
        )
        self.downsample3 = nn.Sequential(
            nn.LayerNorm([256, 4, 12], eps=1e-6),
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
        )
        # 64 512 2 6
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)

        self.stage1 = Block(dim=512)
        self.stage2 = Block(dim=512)
        self.stage3 = Block(dim=512)
        self.stage4 = Block(dim=512)

    def forward(self, x):
        # patched
        x = self.stem(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool1(x)
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


class ConvNeXt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = Extractor()
        self.classifier = Classifier(512, 5994)

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x
