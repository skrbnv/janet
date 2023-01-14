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
        if type(stride) == int:
            self.sh, self.sw = stride, stride
        else:
            self.sh, self.sw = stride[0], stride[1]
        if type(padding) == int:
            self.ph, self.pw = padding, padding
        else:
            self.ph, self.pw = padding[0], padding[1]
        if bias is True:
            self.biases = nn.Parameter(torch.Tensor(planes_out, 1))
            nn.init.kaiming_uniform_(self.biases)
        else:
            self.biases = 0
        self.skip = skip

    def extra_repr(self):
        return f'filters: {tuple(self.filters.shape)}, kernel: {self.kernel_size}, stride: {self.sh, self.sw}, padding: {self.ph, self.pw}, bias: {self.biases != 0}'

    def forward(self, input):
        # input shape (bs, n, h, w)
        ph, pw, sh, sw, ks = self.ph, self.pw, self.sh, self.sw, self.kernel_size
        bs, p_in, p_out = input.size(0), input.size(1), self.filters.size(0)
        device = input.device
        output = torch.empty(
            (bs, p_out, int((input.size(2) + ph * 2 - ks + 1) / sh),
             int((input.size(3) + pw * 2 - ks + 1) / sw))).to(device)
        if self.skip is True:
            return output
        data = F.pad(input, (pw, pw, ph, ph), mode="constant", value=0)
        for y in range(0 + ph, data.size(-2) - ph, sh):
            for x in range(0 + pw, data.size(-1) - pw, sw):
                spot = torch.flatten(data[:, :, y - ph:y + ph + 1,
                                          x - pw:x + pw + 1],
                                     start_dim=2)
                vs = spot.size(-1) + 1
                spot = torch.cat((torch.ones(bs, p_in, 1).to(device), spot),
                                 dim=-1).view(-1, vs)
                slice = torch.einsum("bi,bj->bij", spot,
                                     spot).view(bs, p_in, vs, vs)

                # multiply weights (filters) by matrix with auto-broadcasting matrix
                # bmm = self.filters*mxs

                # broadcast slice   (bs, p_in, dim, dim)    to (bs, p_out, p_in, dim, dim)
                # broadcast filters (p_out, p_in, dim, dim) to (bs, p_out, p_in, dim, dim)
                # multiply between, then sum last three dim
                # add biases if needed
                # return

                output[:, :, int(
                    (y - ph) /
                    sh), int(
                        (x - pw) /
                        sw)] = torch.sum(slice.unsqueeze(1).broadcast_to(
                            bs, p_out, p_in, vs, vs) * self.filters.unsqueeze(
                                0).broadcast_to(bs, p_out, p_in, vs, vs),
                                         dim=(2, 3, 4)) + self.biases

                #for i in range(p_out):
                #    output[:, i, int((y - ph) / sh),
                #           int((x - pw) / sw)] = torch.sum(
                #               (slice * self.filters[i]), dim=(1, 2, 3))
                #    if self.biases is not None:
                #        output += self.biases[i]
        return output


class PolyConv2d(nn.Module):
    def __init__(self,
                 planes_in,
                 planes_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 skip=False,
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
        self.activation = nn.Identity()  # nn.ReLU()
        self.extras = nn.Sequential(
            *extras) if extras is not None or len(extras) == 0 else None

    def forward(self, x):
        x = self.poly(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.extras is not None:
            x = self.extras(x)
        return x


class Extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adjust = PolyConv2d(planes_in=1,
                                 planes_out=16,
                                 kernel_size=7,
                                 stride=(1, 3),
                                 padding=3,
                                 bias=False,
                                 skip=False,
                                 extras=[nn.AvgPool2d((2, 2))])

        seq = [
            PolyConv2d(planes_in=16 * 2**i,
                       planes_out=16 * 2**(i + 1),
                       kernel_size=5,
                       stride=2,
                       padding=2,
                       bias=False,
                       extras=[nn.AvgPool2d((2, 2))]) for i in range(2)
        ]
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
    def __init__(self, midsize=256, num_classes=5994) -> None:
        super().__init__()
        self.extractor = Extractor()
        self.classifier = Classifier(size_in=midsize, size_out=num_classes)

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x
