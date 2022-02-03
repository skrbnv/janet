import torch
import torch.nn as nn
''' V8 WA block after each 2d convolution, either funneled or not '''


class WeightedAttention(nn.Module):
    ''' identity not going to work if h != w '''
    ''' also if h != w matmul output will not match parent convolution '''
    def __init__(self, planes, h, w, residual=False) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(planes, h, h))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(planes)
        self.activation = nn.GELU()
        self.residual = residual
        if self.residual:
            self.weight_identity = nn.Parameter(torch.Tensor([.5]))

    def forward(self, x):
        if self.residual:
            identity = x.clone()
        bs = x.shape[0]
        x = torch.bmm(x.flatten(0, 1), x.flatten(0, 1).transpose(1, 2))
        x = torch.bmm(
            x,
            self.weights.broadcast_to((bs, *self.weights.shape)).flatten(0, 1))
        x = x.view((bs, -1, *x.shape[1:]))
        if self.residual:
            x = (1 -
                 self.weight_identity) * x + self.weight_identity * identity
        x = self.bn(x)
        x = self.activation(x)
        return x


class Conv2dWA(nn.Module):
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
        self.activation = nn.GELU()
        self.extras = nn.Sequential(*extras) if extras is not None else None
        self.wa = WeightedAttention(planes_out, input_shape[0], input_shape[1],
                                    residual)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.extras is not None:
            x = self.extras(x)
        x = self.wa(x)
        return x


class Extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adjust = Conv2dWA(
            input_shape=(64, 192),
            planes_in=1,
            planes_out=128,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
            residual=False,
            extras=[nn.AvgPool2d(3, stride=(1, 3), padding=1)])

        # had to use 128 instead of 64 to prevent division by 0
        seq = [
            Conv2dWA(input_shape=(int(64 / 4**(i + 1)), int(64 / 4**(i + 1))),
                     planes_in=128 * 2**i,
                     planes_out=128 * 2**(i + 1),
                     kernel_size=3,
                     stride=2,
                     padding=1,
                     bias=False,
                     residual=True,
                     extras=[nn.AvgPool2d(2, 2)]) for i in range(3)
        ]
        seq[-1].wa = nn.Identity()
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
