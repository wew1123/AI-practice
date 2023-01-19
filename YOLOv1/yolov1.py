import torch
import numpy as np
from torch.nn import Sequential,functional as F
from torch import nn

num_classes = 5
whxyc = 5
length = num_classes+whxyc*2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input 448*448
        self.layer1 = nn.Sequential(
            # 7*7*64
            # 输入是 3*448*448
            nn.Conv2d(3, 64, 7, stride=2,padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=1),
            # nn.LeakyReLU(),
        )# [64,110,110]
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,192,3),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),
            nn.LeakyReLU(),
        )# [192,54,54]
        self.layer3 = nn.Sequential(
            nn.Conv2d(192,128,1),
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 512, 3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 512, 3),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 512, 3),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 512, 3),
            nn.Conv2d(512, 256, 1,padding=1),
            nn.Conv2d(256, 512, 3,padding=1),
            nn.Conv2d(512, 512, 1,padding=1),
            nn.Conv2d(512, 1024, 3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.Conv2d(512, 1024, 3),
            nn.Conv2d(1024, 512, 1),
            nn.Conv2d(512, 1024, 3,padding=1),
            nn.Conv2d(1024, 1024, 3,padding=1),
            nn.Conv2d(1024,1024,kernel_size=2,stride=2,padding=1),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3,padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3,padding=1),
        )
        self.layer7 = nn.Sequential(
            nn.Linear(1024*7*7,4096),
        )
        self.layer8 = nn.Sequential(
            nn.Linear(4096,7*7*length)
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # [512, 25, 25]
        x = self.layer4(x) # [1024, 7, 7 ]
        x = self.layer5(x) #
        x = self.layer6(x)
        x = x.view(-1,1024*7*7)
        # x = x.view(x.size()[0], -1)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.view(-1,7,7,length)
        return x

x = torch.randn(5, 3,448,448)
net = Net()
y = net(x)
print(y.shape)