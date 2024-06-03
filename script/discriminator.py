import torch
from torch import nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #input: 128 * 128 * 3
        self.local1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #64*64*64
        self.local2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        #32*32*128
        self.local3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        #16*16*256
        self.local4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        #8*8*512
        self.local5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        #4*4*512=8192

        # input: 256*256*3
        self.global1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) #128*128*64
        self.global2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )#32*32*128
        self.global3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )#16*16*256
        self.global4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )#8*8*512
        self.global5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )#4*4*512
        self.global6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )#2*2*512
        self.FC = nn.Linear(8192,1024)
        self.catFC = nn.Linear(2048, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        localx, globalx = x
        localx = self.local1(localx)
        localx = self.local2(localx)
        localx = self.local3(localx)
        localx = self.local4(localx)
        localx = self.local5(localx)
        localx = self.FC(localx.view(-1, 8192))

        globalx = self.global1(globalx)
        globalx = self.global2(globalx)
        globalx = self.global3(globalx)
        globalx = self.global4(globalx)
        globalx = self.global5(globalx)
        globalx = self.global6(globalx)
        globalx = self.FC(globalx.view(-1, 8192))
        out = self.sig(self.catFC(torch.cat((localx, globalx), dim=1)))
        return out


