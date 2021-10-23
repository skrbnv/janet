import torch.nn as nn
import torch.nn.functional as F


class EmbeddingsModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 16, 5, stride=2)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
		self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
		#self.dropout1 = nn.Dropout(p=0.2)
		self.dense1 = nn.Linear(4096, 1024)
		self.dense2 = nn.Linear(1024, 128)
		#self.batchnorm1 = nn.BatchNorm1d(128)

	def forward(self, x):
		#x = x.unsqueeze(1)
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = self.flatten(x)
		#x = self.dropout1(x)
		x = self.dense1(x)
		x = F.relu(x)
		x = self.dense2(x)
		#x = self.batchnorm1(x)
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
