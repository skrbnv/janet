import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as D
import random
import numpy as np

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride = 1)
        self.conv3 = nn.Conv2d(32, 64, 1)
        #self.conv4 = nn.Conv2d(64, 128, 2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dense1 = nn.Linear(36864, 256)
        self.dense2 = nn.Linear(256, 128)
        #self.softmax1 = nn.Softmax(dim=1)
        #self.batchnorm1 = nn.BatchNorm1d(128)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        #x = self.conv4(x)
        #x = F.relu(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        #x = self.softmax1(x)
        #x = self.batchnorm1(x)
        return x
    def flatten(self, x):
        #print(x.shape)
        return x.view(x.shape[0],-1)
        #return np.reshape(ndim_array, (ndim_array.shape[0],ndim_array.shape[1]*ndim_array.shape[2]))
