import torch.nn as nn
import torch


class WeightedMultiplication(nn.Module):
    def __init__(self, planes, h) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(planes, h, h))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(planes)
        #self.activation = nn.GELU()

    def forward(self, x):
        bs = x.shape[0]
        x = torch.bmm(x.flatten(0, 1), x.flatten(0, 1).transpose(1, 2))
        # norm to prevent exploding values
        x = torch.triu(x)
        x = x * self.weights.broadcast_to(
            (bs, *self.weights.shape)).flatten(0, 1)
        x = x.view((bs, -1, *x.shape[1:]))
        x = self.bn(x)
        #x = self.activation(x)
        return x


class Patches(nn.Module):
    def __init__(self, sizes=[]) -> None:
        super().__init__()
        planes_in, planes_out, h_in, h_out = sizes
        self.wm = nn.Parameter(torch.Tensor([.5]))
        self.wr = nn.Parameter(torch.Tensor([.5]))
        #if h_in == h_out:
        #    kernel_size, stride, padding = 1, 1, 0
        #else:
        #    kernel_size, stride, padding = 3, 2, 1
        self.WM = WeightedMultiplication(planes_out, h_out)
        #self.WM2 = WeightedMultiplication(planes_out, h_out)
        #self.preprocess = nn.Conv2d(planes_in,
        #                            planes_out,
        #                            kernel_size=kernel_size,
        #                            stride=stride,
        #                            padding=padding)

    def forward(self, x):
        res = x.clone()
        x = self.WM(x)
        #x = self.preprocess(x)
        #x = self.WM2(x)
        return self.wm * x + self.wr * res
        # in.shape = (bs, N, x, x)
        # first we reshape to bs*N, x, x


def conv3x3(in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=bias)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, sizes=[]):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes=in_planes,
                             out_planes=planes,
                             stride=stride)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes, stride=1)
        self.shortcut = nn.Identity() if in_planes == planes else nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=stride, bias=False)

        self.extra1 = Patches([planes, planes, sizes[0], sizes[1]])
        self.extra2 = Patches([planes, planes, sizes[1], sizes[1]])

    def forward(self, x):
        c = self.bn1(x)
        c = self.relu(c)
        c = self.conv1(c)
        c = self.extra1(c)

        c = self.bn2(c)
        c = self.relu(c)
        if self.dropout is not None:
            c = self.dropout(c)
        c = self.conv2(c)
        c = self.extra2(c)

        c += self.shortcut(x)
        return c


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 32
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor
        nStages = [32, 32 * k, 64 * k, 128 * k]
        #self.conv1 = conv3x3(in_planes=3, out_planes=nStages[0], stride=1)
        self.conv1 = nn.Conv2d(1,
                               32,
                               kernel_size=7,
                               stride=(1, 3),
                               padding=3,
                               bias=False)
        self.bn01 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(2)
        self.layer1 = self.create_layer(wide_basic,
                                        nStages[1],
                                        n,
                                        dropout_rate,
                                        stride=1,
                                        sizes=[32, 32])
        self.layer2 = self.create_layer(wide_basic,
                                        nStages[2],
                                        n,
                                        dropout_rate,
                                        stride=2,
                                        sizes=[32, 16])
        self.layer3 = self.create_layer(wide_basic,
                                        nStages[3],
                                        n,
                                        dropout_rate,
                                        stride=2,
                                        sizes=[16, 8])
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)
        self.pool = nn.AvgPool2d(8)
        self.relu = nn.ReLU()
        self.__init_wb__()

    def __init_wb__(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def create_layer(self, block, planes, num_blocks, dropout_rate, stride,
                     sizes):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for i, stride in enumerate(strides):
            if i == 0:
                layers.append(
                    block(self.in_planes, planes, dropout_rate, stride, sizes))
            else:
                layers.append(
                    block(self.in_planes, planes, dropout_rate, stride,
                          [sizes[1], sizes[1]]))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn01(out)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        #extracted = out.clone()
        out = self.pool(out)
        out = self.linear(out.flatten(1))
        return out  # , extracted


def WideResNet2810(num_classes=10):
    return Wide_ResNet(28, 5, 0.3, num_classes)
