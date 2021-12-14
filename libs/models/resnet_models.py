import torch.nn as nn
import torch
from libs.models.resnet_original import Bottleneck, resnet18, _resnet, BasicBlock
import random


class ResNet18_64x192_kernel5_nomaxpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = resnet18(pretrained=False)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False), )
        self.basemodel.maxpool = nn.ReLU()
        self.basemodel.fc = nn.Linear(512, 128)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet10_64x192_kernel5_nomaxpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet10',
                                 BasicBlock, [1, 1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False), )
        self.basemodel.maxpool = nn.ReLU()
        self.basemodel.fc = nn.Linear(512, 128)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet8_64x192_kernel5_nomaxpool_001(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet10',
                                 BasicBlock, [1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False), )
        self.basemodel.maxpool = nn.ReLU()
        self.basemodel.fc = nn.Linear(256, 128)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet10_64x192_kernel5_nomaxpool_002(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet10',
                                 Bottleneck, [1, 1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False), )
        self.basemodel.maxpool = nn.ReLU()
        self.basemodel.fc = nn.Linear(2048, 128)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet6_64x192_kernel5_nomaxpool_003(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet10',
                                 BasicBlock, [1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False), )
        self.basemodel.maxpool = nn.ReLU()
        self.basemodel.fc = nn.Linear(128, 128)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet8_64x192_kernel5_nomaxpool_004(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet10',
                                 Bottleneck, [1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False), )
        self.basemodel.maxpool = nn.ReLU()
        self.basemodel.fc = nn.Linear(1024, 128)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet10_64x192_patches_kernel5_nomaxpool_005(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet10',
                                 BasicBlock, [1, 1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Conv2d(1,
                                         64,
                                         kernel_size=5,
                                         stride=1,
                                         padding=2,
                                         bias=False)
        self.basemodel.maxpool = nn.ReLU()
        self.basemodel.fc = nn.Linear(512, 256)
        self.dense = nn.Linear(256 * 3, 128)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        step = int(x.shape[3] / 3)
        a = self.basemodel(x[:, :, :, 0:step])
        b = self.basemodel(x[:, :, :, step:step * 2])
        c = self.basemodel(x[:, :, :, step * 2:step * 3])
        x = torch.cat((a, b, c), dim=1)
        return self.dense(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet12_64x192_kernel5_nomaxpool_bottleneck_006(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet12',
                                 Bottleneck, [1, 1, 2, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False), )
        self.basemodel.maxpool = nn.ReLU()
        self.basemodel.fc = nn.Linear(2048, 128)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet10_64x192_kernel5(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet10',
                                 BasicBlock, [1, 1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=(11, 5),
                      stride=(1, 3),
                      padding=(5, 2),
                      bias=False))
        self.basemodel.fc = nn.Linear(512, 630)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet8_64x192_kernel5(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet8',
                                 BasicBlock, [1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False))
        self.basemodel.fc = nn.Linear(256, 630)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet6_64x192_kernel5(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet8',
                                 BasicBlock, [1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False))
        self.basemodel.fc = nn.Linear(128, 630)

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet10_64x192_kernel5_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet10',
                                 BasicBlock, [1, 1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 128)
        self.model = torch.nn.Sequential(
            *(list(self.basemodel.children())[:-2]))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            rand = random.randint(0, x.shape[2] * x.shape[3] - 1)
            mask = torch.zeros(x.shape)
            mask.view(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3])[:, :, rand] = 1
            #mask.view(
            #    x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
            #)[:, :,
            #  [i for i in range(x.shape[2] * x.shape[3]) if i != rand]] = 1
            x = x * mask.to(x.device)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def innerModel(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet18_64x192_kernel5_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet18',
                                 BasicBlock, [2, 2, 2, 2],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 128)
        self.model = torch.nn.Sequential(
            *(list(self.basemodel.children())[:-2]))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            rand = random.randint(0, x.shape[2] * x.shape[3] - 1)
            mask = torch.zeros(x.shape)
            mask.view(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3])[:, :, rand] = 1
            #mask.view(
            #    x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
            #)[:, :,
            #  [i for i in range(x.shape[2] * x.shape[3]) if i != rand]] = 1
            x = x * mask.to(x.device)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def innerModel(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet18_64x192_kernel5_dropouts(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet18',
                                 BasicBlock, [2, 2, 2, 2],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 128)
        self.model = torch.nn.Sequential(
            *(list(self.basemodel.children())[:-2]))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            rands = random.sample(range(x.shape[2] * x.shape[3]), 2)
            mask = torch.zeros(x.shape)
            for rand in rands:
                mask.view(x.shape[0], x.shape[1],
                          x.shape[2] * x.shape[3])[:, :, rand] = 1
            #mask.view(
            #    x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
            #)[:, :,
            #  [i for i in range(x.shape[2] * x.shape[3]) if i != rand]] = 1
            x = x * mask.to(x.device)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def innerModel(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet10_64x192_kernel5_dropouts(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet18',
                                 BasicBlock, [1, 1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 128)
        self.model = torch.nn.Sequential(
            *(list(self.basemodel.children())[:-2]))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            rands = random.sample(range(x.shape[2] * x.shape[3]), 2)
            mask = torch.zeros(x.shape)
            for rand in rands:
                mask.view(x.shape[0], x.shape[1],
                          x.shape[2] * x.shape[3])[:, :, rand] = 1
            #mask.view(
            #    x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
            #)[:, :,
            #  [i for i in range(x.shape[2] * x.shape[3]) if i != rand]] = 1
            x = x * mask.to(x.device)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def innerModel(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet10_64x192_kernel5_4dropouts(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet18',
                                 BasicBlock, [1, 1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 128)
        self.model = torch.nn.Sequential(
            *(list(self.basemodel.children())[:-2]))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            rands = random.sample(range(x.shape[2] * x.shape[3]), 4)
            mask = torch.zeros(x.shape)
            for rand in rands:
                mask.view(x.shape[0], x.shape[1],
                          x.shape[2] * x.shape[3])[:, :, rand] = 1
            #mask.view(
            #    x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
            #)[:, :,
            #  [i for i in range(x.shape[2] * x.shape[3]) if i != rand]] = 1
            x = x * mask.to(x.device)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def innerModel(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet10_64x192_kernel5_8dropouts(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet10',
                                 BasicBlock, [1, 1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 128)
        self.model = torch.nn.Sequential(
            *(list(self.basemodel.children())[:-2]))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            rands = random.sample(range(x.shape[2] * x.shape[3]), 8)
            mask = torch.zeros(x.shape)
            for rand in rands:
                mask.view(x.shape[0], x.shape[1],
                          x.shape[2] * x.shape[3])[:, :, rand] = 1
            #mask.view(
            #    x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
            #)[:, :,
            #  [i for i in range(x.shape[2] * x.shape[3]) if i != rand]] = 1
            x = x * mask.to(x.device)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def innerModel(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet12_64x192_kernel5_8dropouts(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet12',
                                 BasicBlock, [1, 2, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 128)
        self.model = torch.nn.Sequential(
            *(list(self.basemodel.children())[:-2]))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            rands = random.sample(range(x.shape[2] * x.shape[3]), 8)
            mask = torch.zeros(x.shape)
            for rand in rands:
                mask.view(x.shape[0], x.shape[1],
                          x.shape[2] * x.shape[3])[:, :, rand] = 1
            #mask.view(
            #    x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
            #)[:, :,
            #  [i for i in range(x.shape[2] * x.shape[3]) if i != rand]] = 1
            x = x * mask.to(x.device)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def innerModel(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))


class ResNet8_64x192_kernel5_32dropouts(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = _resnet('resnet8',
                                 BasicBlock, [1, 1, 1],
                                 pretrained=False,
                                 progress=True)
        self.basemodel.conv1 = nn.Sequential(
            nn.Conv2d(1,
                      64,
                      kernel_size=5,
                      stride=(1, 3),
                      padding=2,
                      bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 128)
        self.model = torch.nn.Sequential(
            *(list(self.basemodel.children())[:-2]))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            rands = random.sample(range(x.shape[2] * x.shape[3]), 32)
            mask = torch.zeros(x.shape)
            for rand in rands:
                mask.view(x.shape[0], x.shape[1],
                          x.shape[2] * x.shape[3])[:, :, rand] = 1
            #mask.view(
            #    x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
            #)[:, :,
            #  [i for i in range(x.shape[2] * x.shape[3]) if i != rand]] = 1
            x = x * mask.to(x.device)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def innerModel(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))
