'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WeightedMultiplication(nn.Module):
    def __init__(self, planes, h) -> None:
        super().__init__()
        self.div = math.sqrt(h)
        self.weights = nn.Parameter(torch.Tensor(planes, h, h))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(planes)
        #self.activation = nn.GELU()

    def forward(self, x):
        bs = x.shape[0]
        #maxel = torch.max(torch.abs(x))
        x = torch.bmm(x.flatten(0, 1), x.flatten(0, 1).transpose(1, 2))
        # norm to prevent exploding values
        #x /= maxel*self.div
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
        if h_in == h_out:
            kernel_size, stride, padding = 1, 1, 0
        else:
            kernel_size, stride, padding = 3, 2, 1
        self.WM = WeightedMultiplication(planes_in, h_in)
        #self.WM2 = WeightedMultiplication(planes_out, h_out)
        self.preprocess = nn.Conv2d(planes_in,
                                    planes_out,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

    def forward(self, x, conv_val):
        x = self.WM(x)
        x = self.preprocess(x)
        #x = self.WM2(x)
        return self.wm * x + self.wr * conv_val
        # in.shape = (bs, N, x, x)
        # first we reshape to bs*N, x, x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sizes={}):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.extra = Patches(sizes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        res = self.extra(x, out)
        return res


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1,
                               64,
                               kernel_size=7,
                               stride=(1, 3),
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,
                                       64,
                                       num_blocks[0],
                                       stride=1,
                                       sizes=[[64, 64, 64, 64],
                                              [64, 64, 64, 64]])
        self.layer2 = self._make_layer(block,
                                       128,
                                       num_blocks[1],
                                       stride=2,
                                       sizes=[[64, 128, 64, 32],
                                              [128, 128, 32, 32]])
        self.layer3 = self._make_layer(block,
                                       256,
                                       num_blocks[2],
                                       stride=2,
                                       sizes=[[128, 256, 32, 16],
                                              [256, 256, 16, 16]])
        self.layer4 = self._make_layer(block,
                                       512,
                                       num_blocks[3],
                                       stride=2,
                                       sizes=[[256, 512, 16, 8],
                                              [512, 512, 8, 8]])
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, sizes):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, sizes[i]))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
