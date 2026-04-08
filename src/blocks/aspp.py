import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False)
        self.atrous6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.atrous12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.atrous18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.atrous1(x)
        x2 = self.atrous6(x)
        x3 = self.atrous12(x)
        x4 = self.atrous18(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.project(x_cat)
