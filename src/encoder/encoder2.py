import torch
import torch.nn as nn
from ..blocks.conv_block import ConvBlock
from ..blocks.se_block import SEBlock

class Encoder2(nn.Module):
    def __init__(self, in_channels, features=[64,128,256,512]):
        super(Encoder2, self).__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        prev_ch = in_channels
        for f in features:
            self.blocks.append(ConvBlock(prev_ch, f))
            self.se_blocks.append(SEBlock(f))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_ch = f

    def forward(self, x):
        skips = []
        for block, se, pool in zip(self.blocks, self.se_blocks, self.pools):
            x = block(x)
            x = se(x)
            skips.append(x)
            x = pool(x)
        return x, skips
