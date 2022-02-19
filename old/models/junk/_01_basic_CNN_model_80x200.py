import torch.nn as nn
import torch.nn.functional as F


class EmbeddingsModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 16, 5, stride=2)
		self.maxpool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
		self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
		self.dropout1 = nn.Dropout(p=0.5)
		self.dense1 = nn.Linear(1152, 1024)
		self.dense2 = nn.Linear(1024, 256)
		self.bn1 = nn.BatchNorm1d(256)
		self.dense3 = nn.Linear(256, 128)

	def forward(self, x):
		#x = x.unsqueeze(1)
		x = self.conv1(x)
		x = F.relu(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.maxpool(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = F.relu(x)
		x = self.dense2(x)
		x = F.relu(x)
		x = self.bn1(x)
		x = self.dense3(x)
		x = F.normalize(x)
		return x

	def flatten(self, x):
		return x.view(x.shape[0], -1)


class TripletModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.basemodel = EmbeddingsModel()

	def forward(self, x, y, z):
		anchor = self.innerModel(x)
		positive = self.innerModel(y)
		negative = self.innerModel(z)
		return anchor, positive, negative

	def innerModel(self, x):
		return self.basemodel(x)
