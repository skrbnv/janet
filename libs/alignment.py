import torch
import torch.nn as nn


class DirectionalConv2d(nn.Module):
    def __init__(self,
                 shape: tuple,
                 in_channels: int,
                 out_channels: int,
                 kernel: int | tuple,
                 stride: int | tuple = 1,
                 padding: int | tuple = 0,
                 groups: int = 1,
                 bias: bool = True,
                 extras: list | None = None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.extras = nn.Sequential(*extras) if extras is not None else None
        self.alignment = DirectionalAlignment(out_channels, shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.extras is not None:
            x = self.extras(x)
        x = self.alignment(x)
        return x


class DirectionalAlignment(nn.Module):
    def __init__(self, planes: int, shape: tuple) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            torch.Tensor(planes, shape[0], shape[0]))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(planes)
        self.dim = torch.sqrt(torch.tensor(shape[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        x = x @ x.transpose(-1, -2)
        x /= self.dim
        x *= self.weights
        x = torch.sum(x, dim=-1, keepdim=True)
        x = identity + x
        x = self.bn(x)
        return x


class MultiDirectionalAlignment(nn.Module):
    def __init__(self, planes_multiplier: int, planes: int, shape: tuple) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            torch.Tensor(planes_multiplier, planes, shape[0], shape[0]))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(planes*planes_multiplier)
        self.dim = torch.sqrt(torch.tensor(shape[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        x = x @ x.transpose(-1, -2)
        x /= self.dim
        x = x.unsqueeze(1).expand(-1, self.weights.shape[0], -1, -1, -1)*self.weights
        x = torch.sum(x, dim=-1, keepdim=True)
        x = identity.unsqueeze(1) + x
        x = x.flatten(1, 2)
        x = self.bn(x)
        # x = torch.nn.functional.relu(x)
        return x


class PlanarConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: int | tuple = 3,
                 stride: int | tuple = 1,
                 padding: int | tuple = 1,
                 pooling: list = [],
                 shape=None,
                 extras: list | None = None) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding), nn.BatchNorm2d(out_channels),
            nn.GELU(approximate='tanh'))
        self.alignment = PlanarAlignment(out_channels, pooling)
        self.extras = nn.Sequential(*extras) if extras is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.extras is not None:
            x = self.extras(x)
        x = self.alignment(x)
        return x


class PlanarAlignment(nn.Module):
    def __init__(self, planes: int, pooling: list = []) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(planes, planes))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(planes)
        self.pool = nn.Identity() if len(pooling) == 0 else nn.Sequential(
            *pooling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        # shape is B,C,H,W
        x = self.pool(x)
        x = x.flatten(-2)
        # shape is B,C,H*W
        x = x @ x.transpose(-1, -2)
        scale = torch.var(x) / torch.var(identity)
        x = x / scale
        # shape is B,C,C
        x *= self.weights
        # shape is B,C,C
        x = torch.sum(x, dim=-1, keepdim=True)
        # shape is B,C,1
        x = x.unsqueeze(-1)
        # shape is B,C,1,1
        x = identity + x  # auto broadcast to identity
        x = self.bn(x)
        return x
