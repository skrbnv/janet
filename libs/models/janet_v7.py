import torch
import torch.nn as nn


class AttentionWithSimpleConv(nn.Module):
    def __init__(self, num_heads) -> None:
        super().__init__()
        self.multilinear1 = nn.Conv2d(num_heads,
                                      num_heads,
                                      kernel_size=1,
                                      groups=num_heads)
        self.bn = nn.BatchNorm2d(num_heads)
        self.relu = nn.ReLU()

    def forward(self, x):
        shape = x.shape
        x = torch.bmm(x.flatten(0, 1), x.flatten(0, 1).transpose(1, 2))
        x = x.view(shape)
        x = self.multilinear1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class WeightedAttention(nn.Module):
    def __init__(self, num_heads, h, w) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(num_heads, h, h))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(num_heads)
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.shape[0]
        x = torch.bmm(x.flatten(0, 1), x.flatten(0, 1).transpose(1, 2))
        x = torch.bmm(
            x,
            self.weights.broadcast_to((bs, *self.weights.shape)).flatten(0, 1))
        x = x.view((bs, -1, *x.shape[1:]))
        x = self.bn(x)
        x = self.relu(x)
        return x


class AttentionWithFullConv(nn.Module):
    def __init__(self, num_heads, h, w) -> None:
        super().__init__()
        self.multilinear1 = nn.Conv1d(num_heads * h * w,
                                      num_heads * h * w,
                                      kernel_size=1,
                                      groups=num_heads)
        self.bn = nn.BatchNorm2d(num_heads)
        self.relu = nn.ReLU()

    def forward(self, x):
        shape = x.shape
        x = torch.bmm(x.flatten(0, 1), x.flatten(0, 1).transpose(1, 2))
        x = self.multilinear1(x.view(shape).flatten(1, 3).unsqueeze(2))
        x = x.view(shape)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,
                               64,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.avgpool1 = nn.AvgPool2d(3, stride=(1, 3), padding=1)
        self.attention = WeightedAttention(64, 64, 64)
        self.stack = self.create_stack(planes_in=64, multiplyer=2, steps=4)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

    def create_block(self, planes_in, planes_out):
        return nn.Sequential(*[
            nn.Conv2d(planes_in,
                      planes_out,
                      kernel_size=5,
                      stride=2,
                      padding=2,
                      bias=False),
            nn.BatchNorm2d(planes_out),
            nn.ReLU()
        ])

    def create_stack(self, planes_in=64, multiplyer=2, steps=4):
        return nn.Sequential(*[
            self.create_block(planes_in *
                              multiplyer**(i), planes_in * multiplyer**(i + 1))
            for i in range(steps)
        ])

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.avgpool1(x)
        x = self.attention(x)
        # x = x.mean(dim=1, keepdim=False).flatten(start_dim=1)
        x = self.stack(x)
        x = self.avgpool2(x)
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


class Janet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = Extractor()
        self.classifier = Classifier(size_in=1024, size_out=630)

    def forward(self, x):
        x = self.extractor(x)

        # (bs,num_heads,h,w)->(bs,1,h,w)->(bs,h,w)->(bs,h*w)
        x = self.classifier(x)
        return x
