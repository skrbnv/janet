import torch.nn as nn
import torch.nn.functional as F

# bottlenecks > equal?

class EmbeddingsModel(nn.Module):

	def __init__(self):
		super().__init__()
		# (64,1,80,200)
		self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,5), stride=(1,2), padding=(1,2), bias=False)
		# (64,32,80,100)
		self.bn32 = nn.BatchNorm2d(32)
		# (64,32,80,100)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False)
		# (64,64,40,50)
		self.bn64 = nn.BatchNorm2d(64)
		# (64,64,40,50)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
		# (64,128,20,25)
		self.bn128 = nn.BatchNorm2d(128)
		# (64,128,20,50)
		self.block128 = nn.Sequential(
            nn.Conv2d(128, 64, 1,stride=1, padding=0, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			# (64,128,20,25)
			nn.Conv2d(64, 128, 1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(128),
			# (64,128,20,25)
		)
		# (64,128,20,25)
		self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=(2,1), bias=False)
		# (64,256,10,12)
		self.bn256 = nn.BatchNorm2d(256)
		# (64,256,10,12)

		self.block256 = nn.Sequential(
            nn.Conv2d(256, 64, 1,stride=1, padding=0, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			# (64,256,10,12)
			nn.Conv2d(64, 256, 1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(256),
			# (64,256,10,12)
		)
		# (64,256,10,12)
		self.conv5 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
		self.bn512 = nn.BatchNorm2d(512)
		# (64,512,5,6)

		self.block512 = nn.Sequential(
            nn.Conv2d(512, 64, 1,stride=1, padding=0, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			# (64,512,5,6)
			nn.Conv2d(64, 512, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
			# (64,512,5,6)
		)
		# (64,512,5,6)
		self.conv6 = nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=(1,2), bias=False)

		self.bn1024 = nn.BatchNorm2d(1024)
		# (64,1024,2,3)

		self.avgpool310 = nn.AvgPool2d(3, padding=(1,0))
		# (64,1024,1,1)

		self.fc1 = nn.Linear(1024, 512, bias=False)
		self.bn1024f = nn.BatchNorm1d(512)
		#self.dfc1 = nn.Dropout(p=0.1)
		self.fc2 = nn.Linear(512, 128)

	def forward(self, x):
		#x = x.unsqueeze(1)
		x = self.conv1(x)
		x = self.bn32(x)
		x = F.relu(x)
        # (64,32,80,100)
		x = self.conv2(x)
		x = self.bn64(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = self.bn128(x)
		x = F.relu(x)
		# (64,128,20,25)
		for i in range(3):
			identity = x
			x = self.block128(x)
			x += identity
			x = F.relu(x)
		# (64,128,20,25)
		x = self.conv4(x)
		x = self.bn256(x)
		x = F.relu(x)
		# (64,256,10,12)
		for i in range(3):
			identity = x
			x = self.block256(x)
			x += identity
			x = F.relu(x)
		# (64,256,10,12)
		x = self.conv5(x)
		x = self.bn512(x)
		x = F.relu(x)
		# (64,512,5,6)
		for i in range(3):
			identity = x
			x = self.block512(x)
			x += identity
			x = F.relu(x)
		# (64,512,5,6)
		x = self.conv6(x)
		x = self.bn1024(x)
		x = F.relu(x)
		# (64,1024,2,3)
		x = self.avgpool310(x)
		# (64,1024,1,1)
		x = self.flatten(x)
		# (64,1024)
		x = self.fc1(x)
		# (64,512)
		x = self.bn1024f(x)
		#x = self.dfc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		# (64,128)
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
