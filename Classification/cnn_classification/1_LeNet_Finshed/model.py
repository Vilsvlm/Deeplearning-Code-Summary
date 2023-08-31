import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__ (self):
        super(LeNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*5*5,120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84,10)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.classifier(x)
        return x


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 5)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
#         x = self.pool1(x)            # output(16, 14, 14)
#         x = F.relu(self.conv2(x))    # output(32, 10, 10)
#         x = self.pool2(x)            # output(32, 5, 5)
#         x = x.view(-1, 32*5*5)       # output(32*5*5)


#         x = F.relu(self.fc1(x))      # output(120)
#         x = F.relu(self.fc2(x))      # output(84)
#         x = self.fc3(x)              # output(10)
#         return x


inp = torch.rand(32,3,32,32)
model = LeNet()
print(model)