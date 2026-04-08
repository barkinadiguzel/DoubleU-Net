import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(SegmentationHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)
