import os
import random
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn

import torch.nn.functional as F
import cv2
import numpy as np
import torch.optim as optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips


import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    # conv keep resolution
    # Pooling downsamples
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    # in_ch = 4 (masked_img + mask)
    def __init__(self, in_ch=4, base=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(in_ch, base)            
        self.enc2 = ConvBlock(base, base * 2)         
        self.enc3 = ConvBlock(base * 2, base * 4)     
        self.enc4 = ConvBlock(base * 4, base * 8)     
        self.bottleneck = ConvBlock(base * 8, base * 8)  

        self.up4 = nn.ConvTranspose2d(base * 8, base * 8, 2, 2)
        self.dec4 = ConvBlock(base * 8 + base * 8, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, 2)
        self.dec3 = ConvBlock(base * 4 + base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, 2)
        self.dec2 = ConvBlock(base * 2 + base * 2, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, 2)
        self.dec1 = ConvBlock(base + base, base)

        self.final_conv = nn.Conv2d(base, 3, kernel_size=1)
        self.final_act = nn.Sigmoid()

    def forward(self, masked_img, mask):
        # add mask as a channel
        x = torch.cat([masked_img, mask], dim=1)  # [B, 4, H, W]

        e1 = self.enc1(x)                   
        e2 = self.enc2(self.pool(e1))       
        e3 = self.enc3(self.pool(e2))       
        e4 = self.enc4(self.pool(e3))       
        b = self.bottleneck(self.pool(e4))  


        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))  # 32        
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # 64
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # 128
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # 256

        return self.final_act(self.final_conv(d1))