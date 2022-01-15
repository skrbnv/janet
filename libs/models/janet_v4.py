import torch
import torch.nn as nn
import math


class MyLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(
            weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w_times_x = torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)


class Attention(nn.Module):
    def __init__(self, num_heads, h, w) -> None:
        super().__init__()
        # instead of nn.Linear(size**2, size**2)
        # we do Conv2d with kernel_size=1 grouped into num_heads groups
        # 1x1 convolution when out_filters = in_filters is pretty much fully connected layer,
        # and groups = in_filters is parallelizing computations per head
        self.weights = nn.Parameter(torch.Tensor(num_heads, h, h))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(num_heads)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (bs, num_heads, h, w)
        bs = x.shape[0]
        x = torch.bmm(x.flatten(0, 1), x.flatten(0, 1).transpose(1, 2))
        # (bs*num_heads, h, h) or (bs*num_heads, w, w) - depends on transpose
        x = torch.bmm(
            x,
            self.weights.broadcast_to((bs, *self.weights.shape)).flatten(0, 1))
        # shape here is still (num_heads, size, size)
        # now we average over heads, kind of like avg pooling but across planes
        x = x.view((bs, -1, *x.shape[1:]))
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
        self.attention = Attention(num_heads, 64, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool1(x)
        x = self.attention(x)
        x = x.mean(dim=1, keepdim=False).flatten(start_dim=1)
        return x


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
        self.extractor = Extractor(num_heads=4)
        self.classifier = Classifier(size_in=4096,
                                     size_out=630,
                                     midsize=1024,
                                     dropout=.5)

    def forward(self, x):
        x = self.extractor(x)

        # (bs,num_heads,h,w)->(bs,1,h,w)->(bs,h,w)->(bs,h*w)
        x = self.classifier(x)
        return x
