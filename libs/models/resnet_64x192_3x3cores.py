#from torch import Tensor, vstack
import torch.nn as nn
#import torch.nn.functional as F
#from torchvision.models import resnet34
from libs.custom_resnet import resnet50
import torch
import torch.nn.functional as F


###  ^^^^^^^^^^ CUSTOM RESNET ^^^^^^^^^^
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.basemodel = resnet50(pretrained=False)
        # 64x192x1

        # 64x64x64
        #nn.BatchNorm2d(64),
        #nn.MaxPool2d(kernel_size=(1, 2)),
        # 48x48x32
        #nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
        #nn.BatchNorm2d(64),
        #nn.ReLU(),
        # 24x24x64
        #nn.MaxPool2d(2)

        #self.basemodel.fc = nn.Sequential(nn.Linear(2048, 1024),
        #                                  nn.Dropout(.5), nn.ReLU(),
        #                                  nn.Linear(1024, 256))

    def forward(self, n):
        n = self.innerModel(n)
        return n

    def innerModel(self, x):
        return self.basemodel(x)
        # .unsqueeze(1).broadcast_to(x.shape[0], 3, x.shape[1], x.shape[2]))
