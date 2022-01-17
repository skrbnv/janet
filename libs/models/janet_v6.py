import torch
import torch.nn as nn


class AttentionWithSimpleConv(nn.Module):
    def __init__(self, num_heads) -> None:
        super().__init__()
        self.multilinear1 = nn.Conv2d(num_heads, num_heads, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_heads)
        self.relu = nn.ReLU()

    def forward(self, x):
        shape = x.shape
        x = torch.bmm(x.flatten(start_dim=0, end_dim=1),
                      x.flatten(start_dim=0, end_dim=1).transpose(1, 2))
        x = x.view(shape)
        x = self.multilinear1(x)
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
        x = torch.bmm(x.flatten(start_dim=0, end_dim=1),
                      x.flatten(start_dim=0, end_dim=1).transpose(1, 2))
        x = self.multilinear1(
            x.view(shape).flatten(start_dim=1, end_dim=3).unsqueeze(2))
        x = x.view(shape)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Extractor(nn.Module):
    def __init__(self, num_heads=8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,
                               num_heads,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_heads)
        self.avgpool1 = nn.AvgPool2d(3, stride=(1, 3), padding=1)
        self.attn1 = AttentionWithSimpleConv(num_heads, 64, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool1(x)
        x = self.attn1(x)
        x = x.mean(dim=1, keepdim=False).flatten(start_dim=1)
        return x


class ExtractorOld(nn.Module):
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
        self.conv2 = nn.Conv2d(64,
                               128,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,
                               256,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,
                               512,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512,
                               1024,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.avgpool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.avgpool2(x)
        return x.flatten(1)


class Classifier(nn.Module):
    def __init__(self, size_in, size_out, midsize=1024, dropout=.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(size_in, midsize)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(midsize, size_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ClassifierOld(nn.Module):
    def __init__(self, size_in, size_out) -> None:
        super().__init__()
        self.linear1 = nn.Linear(size_in, size_out)

    def forward(self, x):
        x = self.linear1(x)
        return x


class Janet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = ExtractorOld()
        self.classifier = ClassifierOld(size_in=1024, size_out=630)

    def forward(self, x):
        x = self.extractor(x)

        # (bs,num_heads,h,w)->(bs,1,h,w)->(bs,h,w)->(bs,h*w)
        x = self.classifier(x)
        return x
