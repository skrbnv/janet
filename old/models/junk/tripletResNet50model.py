import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import libs.models.resnet as resnet

ResNetFCOutputSize = 128


class EmbeddingsModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.resnet = resnet.resnet34(pretrained=False)
		self.resnet.fc = nn.Linear(
		    10752, ResNetFCOutputSize
		)    # 512 * num_blocks: 1 for resnet34, 4 for resnet50, ? for resnet 101
		self.resnet.conv1 = nn.Conv2d(1,
		                              64,
		                              kernel_size=7,
		                              stride=2,
		                              padding=3,
		                              bias=False)

	def forward(self, x):
		x = self.resnet(x)
		return x


class TripletModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.basemodel = EmbeddingsModel()

		#self.bn1 = nn.BatchNorm1d(num_features=128)
		#self.fc1 = nn.Linear(ResNetFCOutputSize, 128)

	def forward(self, x, y, z):
		x = self.innerModel(x)
		y = self.innerModel(y)
		z = self.innerModel(z)
		return x, y, z

	def innerModel(self, x):
		x = self.basemodel(x)
		return x