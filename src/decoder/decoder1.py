import torch
import torch.nn as nn
from ..blocks.upsample import UpSample
from ..blocks.conv_block import ConvBlock
from ..blocks.se_block import SEBlock

class Decoder1(nn.Module):
    def __init__(self, features=[512,256,128,64]):
        super(Decoder1, self).__init__()
        self.up = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.se = nn.ModuleList()

        for f in features:
            self.up.append(UpSample(scale_factor=2))
            self.conv.append(ConvBlock(f*2, f))  
            self.se.append(SEBlock(f))

    def forward(self, x, skips):
        for i in range(len(self.up)):
            x = self.up[i](x)
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = self.conv[i](x)
            x = self.se[i](x)
        return x
