import torch
import torch.nn as nn
from torchvision import models

class VGG19Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG19Encoder, self).__init__()
        vgg = models.vgg19(pretrained=pretrained).features
        self.stage1 = nn.Sequential(*vgg[:4])   
        self.stage2 = nn.Sequential(*vgg[4:9])  
        self.stage3 = nn.Sequential(*vgg[9:18]) 
        self.stage4 = nn.Sequential(*vgg[18:27])
        self.stage5 = nn.Sequential(*vgg[27:36])

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return s1, s2, s3, s4, s5
