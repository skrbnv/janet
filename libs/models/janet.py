import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, num_heads, h, w) -> None:
        super().__init__()
        # instead of nn.Linear(size**2, size**2)
        # we do Conv2d with kernel_size=1 grouped into num_heads groups
        # 1x1 convolution when out_filters = in_filters is pretty much fully connected layer,
        # and groups = in_filters is parallelizing computations per head
        '''
        self.multilinear1 = nn.Conv1d(num_heads * h * w,
                                      num_heads * h * w,
                                      kernel_size=1,
                                      groups=num_heads)
        '''
        self.multilinear1 = nn.Conv2d(num_heads, num_heads, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_heads)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input is (num_heads, size, size)
        # using batched self-multiply
        shape = x.shape
        x = torch.bmm(x.flatten(start_dim=0, end_dim=1),
                      x.flatten(start_dim=0, end_dim=1).transpose(1, 2))
        # shape here is still (num_heads, size, size)
        '''
        x = self.multilinear1(
            x.view(shape).flatten(start_dim=1, end_dim=3).unsqueeze(2))
        '''
        # shape here is still (num_heads, size, size)
        # now we average over heads, kind of like avg pooling but across planes
        x = x.view(shape)
        x = self.multilinear1(x)
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
        self.attn1 = Attention(num_heads, 64, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool1(x)
        x = self.attn1(x)
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
