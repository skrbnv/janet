import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, num_heads) -> None:
        super().__init__()
        # instead of nn.Linear(size**2, size**2)
        # we do Conv2d with kernel_size=1 grouped into num_heads groups
        # 1x1 convolution when out_filters = in_filters is pretty much fully connected layer,
        # and groups = in_filters is parallelizing computations per head
        self.linear = nn.Conv2d(1, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input is (num_heads, size, size)
        # using batched self-multiply
        # shape = x.shape
        x = torch.bmm(x.flatten(start_dim=0, end_dim=1),
                      x.flatten(start_dim=0,
                                end_dim=1).transpose(1, 2)).unsqueeze(1)
        # shape here is still (num_heads, size, size)
        x = self.linear(x.unsqueeze(1))
        x = self.bn(x)
        x = self.relu(x)
        # shape here is still (num_heads, size, size)
        # now we average over heads, kind of like avg pooling but across planes

        return x


class Extractor(nn.Module):
    def __init__(self, num_heads=8) -> None:
        super().__init__()
        self.attn1 = Attention(num_heads)
        self.conv1 = nn.Conv2d(1,
                               64,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        #self.avgpool1 = nn.AvgPool2d(3, stride=3, padding=1)
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
        self.avgpool2 = nn.AvgPool2d(4)

    def forward(self, x):
        x = self.attn1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.avgpool2(x)
        return x.flatten(start_dim=1)


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


class Janet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = Extractor(num_heads=3)
        self.classifier = Classifier(size_in=1024,
                                     size_out=630,
                                     midsize=1024,
                                     dropout=.5)

    def forward(self, x):
        x = self.extractor(x)

        # (bs,num_heads,h,w)->(bs,1,h,w)->(bs,h,w)->(bs,h*w)
        x = self.classifier(x)
        return x
