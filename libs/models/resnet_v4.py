import torch.nn as nn
import torch.nn.functional as F

# block64 - 1x1 -> 3x3
# moved ReLU


class EmbeddingsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # (64,1,80,200)
        self.conv1 = nn.Conv2d(1,
                               32,
                               kernel_size=(3, 5),
                               stride=(1, 2),
                               padding=(1, 2))
        # (64,32,80,100)
        self.bn32 = nn.BatchNorm2d(32)
        # (64,32,80,100)
        self.maxpool200 = nn.MaxPool2d(2, padding=(0, 0))
        # (64,32,40,50)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        # (64,64,20,25)
        self.bn64 = nn.BatchNorm2d(64)
        # (64,64,20,25)
        self.conv3 = nn.Conv2d(64,
                               128,
                               kernel_size=5,
                               stride=2,
                               padding=(2, 1))
        # (64,128,10,12)
        self.bn128 = nn.BatchNorm2d(128)
        # (64,128,10,12)
        self.block128 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (64,128,10,12)
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # (64,128,10,12)
        )
        # (64,128,10,12)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        # (64,256,5,6)
        self.bn256 = nn.BatchNorm2d(256)
        # (64,256,5,6)

        self.block256 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # (64,128,10,12)
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            # (64,128,10,12)
        )

        self.maxpool310 = nn.MaxPool2d(3, padding=(1, 0))
        # (64,256,2,2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # (64,512,1,1)
        self.bn512 = nn.BatchNorm2d(512)
        # (64,512,1,1)
        #self.dropout1 = nn.Dropout(p=0.5)
        self.dense1 = nn.Linear(512, 630)

    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn32(x)
        x = F.relu(x)
        x = self.maxpool200(x)
        x = self.conv2(x)
        x = self.bn64(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn128(x)
        x = F.relu(x)
        # (64,128,10,12)
        for i in range(3):
            identity = x
            x = self.block128(x)
            x += identity
            x = F.relu(x)
        # (64,128,10,12)
        x = self.conv4(x)
        x = self.bn256(x)
        x = F.relu(x)
        # (64,256,5,6)
        for i in range(3):
            identity = x
            x = self.block256(x)
            x += identity
            x = F.relu(x)
        # (64,256,5,6)
        x = self.maxpool310(x)
        # (64,256,2,2)
        x = self.conv5(x)
        x = self.bn512(x)
        x = F.relu(x)
        # (64,512,1,1)
        x = self.flatten(x)
        x = self.dense1(x)
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
