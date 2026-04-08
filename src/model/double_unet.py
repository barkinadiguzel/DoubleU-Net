import torch
import torch.nn as nn
from ..encoder.encoder1 import Encoder1
from ..encoder.encoder2 import Encoder2
from ..decoder.decoder1 import Decoder1
from ..decoder.decoder2 import Decoder2
from ..head.heads import SegmentationHead
from ..fusion.multiply import multiply_input_mask

class DoubleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(DoubleUNet, self).__init__()
        # NETWORK 1
        self.encoder1 = Encoder1()
        self.decoder1 = Decoder1()
        self.head1 = SegmentationHead(64, out_channels)

        # NETWORK 2
        self.encoder2 = Encoder2(in_channels)
        self.decoder2 = Decoder2()
        self.head2 = SegmentationHead(64, out_channels)

    def forward(self, x):
        # NETWORK 1
        skips1 = self.encoder1(x)
        d1 = self.decoder1(skips1[-1], skips1[:-1])
        mask1 = self.head1(d1)

        # Fusion
        fused_input = multiply_input_mask(x, mask1)

        # NETWORK 2
        enc2_out, skips2 = self.encoder2(fused_input)
        d2 = self.decoder2(enc2_out, skips1[:-1], skips2)
        mask2 = self.head2(d2)

        return mask1, mask2
