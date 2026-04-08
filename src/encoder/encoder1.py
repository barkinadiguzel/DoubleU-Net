import torch
import torch.nn as nn
from ..blocks.vgg19 import VGG19Encoder

class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.vgg = VGG19Encoder(pretrained=True)

    def forward(self, x):
        s1, s2, s3, s4, s5 = self.vgg(x)
        return [s1, s2, s3, s4, s5]
