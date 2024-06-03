import torch
from torch import nn
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # input_shape: (batch, channel, H, W)
        # 64*64
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=16, padding=16),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )
        self.layer13 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )
        self.layer15 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )
        self.layer16 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32)
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x_masked):
        out = self.layer1(x_masked)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.out(out)
        return out