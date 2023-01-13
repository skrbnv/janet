import torch
import torch.nn as nn
import torch.nn.functional as F


class Poly2d(nn.Module):
    def __init__(
            self,
            planes_in,
            planes_out,
            kernel_size=3,  # only single values allowed rn
            stride=1,
            padding=1,
            bias=True,
            skip=False):
        super(Poly2d, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size cannot be even'
        # create filters
        # |a b c|
        # |d e f| - for kernel size 3x3 we'll have filter size (depth x 3*3+1 x 3*3+1) - due to autocorellation matrix
        # |g h k|
        filters = torch.Tensor(planes_out, planes_in, kernel_size**2 + 1,
                               kernel_size**2 + 1)
        self.filters = nn.Parameter(filters)
        nn.init.kaiming_uniform_(self.filters)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if bias is True:
            self.biases = nn.Parameter(torch.Tensor(planes_out, 1))
            nn.init.kaiming_uniform_(self.biases)
        else:
            self.biases = None
        self.skip = skip

    def forward(self, input):
        # input shape (bs, n, h, w)
        p, s = self.padding, self.stride
        device = input.device
        output = torch.empty(
            (input.size(0), self.filters.size(0),
             int((input.size(2) + p * 2 - self.kernel_size + 1) / self.stride),
             int((input.size(3) + p * 2 - self.kernel_size + 1) /
                 self.stride))).to(device)
        if self.skip is True:
            return output
        data = F.pad(input, (p, p, p, p), mode="constant", value=0)
        for y in range(0 + p, data.size(-2) - p, self.stride):
            for x in range(0 + p, data.size(-1) - p, self.stride):
                spot = torch.flatten(data[:, :, y - p:y + p + 1,
                                          x - p:x + p + 1],
                                     start_dim=2)
                spot = torch.cat((torch.ones(spot.size(0), spot.size(1),
                                             1).to(device), spot),
                                 dim=-1)

                slice = torch.empty((input.size(0), spot.size(1), spot.size(2),
                                     spot.size(2))).to(device)
                for num in range(spot.size(0)):
                    for plane in range(spot.size(1)):
                        slice[num,
                              plane] = torch.outer(spot[num, plane],
                                                   spot[num, plane])

                # multiply weights (filters) by matrix with auto-broadcasting matrix
                # bmm = self.filters*mxs
                for i in range(self.filters.size(0)):
                    output[:, i, int((y - p) / s),
                           int((x - p) / s)] = torch.sum(
                               (self.filters[i] * slice), dim=(1, 2, 3))
                    if self.biases is not None:
                        output += self.biases[i]
        return output


class PolyConv2d(nn.Module):
    def __init__(self,
                 input_shape,
                 planes_in,
                 planes_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True,
                 skip=False,
                 residual=False,
                 extras=None) -> None:
        super().__init__()
        self.poly = Poly2d(planes_in,
                           planes_out,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=bias,
                           skip=skip)
        self.bn = nn.BatchNorm2d(planes_out)
        self.activation = nn.ReLU()
        self.extras = nn.Sequential(*extras) if extras is not None else None
        #self.wm = WeightedMultiplication(planes_out, input_shape[0],
        #                                 input_shape[1], residual)

    def forward(self, x):
        x = self.poly(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.extras is not None:
            x = self.extras(x)
        #x = self.wm(x)
        return x


class Extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adjust = PolyConv2d(
            input_shape=(64, 192),
            planes_in=3,
            planes_out=128,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
            residual=False,
            skip=False,
            extras=[nn.AvgPool2d(3, stride=(1, 3), padding=1)])

        seq = [
            PolyConv2d(input_shape=(int(64 / 4**(i + 1)),
                                    int(64 / 4**(i + 1))),
                       planes_in=128 * 2**i,
                       planes_out=128 * 2**(i + 1),
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=False,
                       residual=True,
                       extras=[nn.AvgPool2d(2, 2)]) for i in range(3)
        ]
        seq[-1].wm = nn.Identity()
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
        self.extractor = Extractor()
        self.classifier = Classifier(size_in=midsize, size_out=num_classes)

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x
