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


def combined_inpainting_loss(
    pred, target, mask, lpips_model,
    weight_hole=60.0,      # l1 hole loss weight
    weight_valid=0.5,      # l1 valid loss weight
    weight_hole_l2=20.0,   # l2 loss in hole
    weight_perc=0.15,      # focus on perceptual loss
):
    # pixel by pixel loss with reconstruction (pred vs target)
    l1 = F.l1_loss(pred, target, reduction='none')
    l2 = F.mse_loss(pred, target, reduction='none')

    # divvision by zero protection
    # get hole and valid pixels sums (how many pixels are in hole and valid)
    hole_pixels = mask.sum() + 1e-8
    valid_pixels = (1.0 - mask).sum() + 1e-8

    # hadamard product (element-wise multiplication)
    hole_loss = (l1 * mask).sum() / hole_pixels # l1 loss only on hole
    valid_loss = (l1 * (1.0 - mask)).sum() / valid_pixels # l1 loss only on valid pixels
    hole_l2 = (l2 * mask).sum() / hole_pixels   # mse loss only on hole

    pixel_loss = (weight_hole * hole_loss) + (weight_valid * valid_loss) + (weight_hole_l2 * hole_l2)
    # get reconstruction from pred and background from target
    # weonly judge the recosntruction
    comp = target * (1.0 - mask) + pred * mask
    total = pixel_loss

 
    # scale to [-1, 1] for lpips
    comp_scaled = comp * 2.0 - 1.0
    target_scaled = target * 2.0 - 1.0
    perc_loss = lpips_model(comp_scaled, target_scaled).mean()
    
    total = total + weight_perc * perc_loss


    return total