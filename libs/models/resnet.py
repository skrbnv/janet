from torch import Tensor, vstack
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = resnet50(pretrained=True)
        #self.basemodel.conv1 = nn.Sequential(
        #    nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
        #    # (128x128)
        #    nn.BatchNorm2d(32),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=(1, 2)),
        #    nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
        #    nn.BatchNorm2d(64),
        #    nn.ReLU(),
        #    #nn.MaxPool2d(2)
        #)
        # (128x128)
        #self.basemodel.fc = nn.Sequential(nn.Linear(2048, 1024),
        #                                  nn.Dropout(.5), nn.ReLU(),
        #                                  nn.Linear(1024, 256))
        #self.basemodel.fc = nn.Linear(2048, 256)

    def forward(self, p, n):
        p = self.innerModel(p)
        n = self.innerModel(n)
        return p, n

    def innerModel(self, x):
        return self.basemodel(
            x.unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))
