from torch import nn
import torch.nn.functional as F

import math

class BaseNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        num_stages = stages_for(input_size)

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        layers = []

        for _ in range(num_stages):
            layers.extend([
                nn.ConvTranspose2d(
                    32, 32,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(inplace=True),
            ])

        layers.append(nn.Conv2d(32, 3, kernel_size=3, padding=1))

        self.upsampling = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return self.upsampling(x)


class Netv2(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        num_stages = stages_for(input_size)

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        layers = []

        for _ in range(num_stages):
            layers.extend([
                nn.ConvTranspose2d(
                    32, 32,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])

        layers.append(nn.Conv2d(32, 3, kernel_size=3, padding=1))

        self.upsampling = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return self.upsampling(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = F.relu(self.conv1(x))
        residual = self.conv2(residual)

        return x + residual        

class Netv3(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        num_stages = stages_for(input_size)

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.refinement = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
        )

        layers = []

        for _ in range(num_stages):
            layers.extend([
                nn.Conv2d(32, 32 * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ])

        layers.append(nn.Conv2d(32, 3, kernel_size=3, padding=1))

        self.upsampling = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.refinement(x)
        return self.upsampling(x)

def stages_for(input_size):
    if input_size not in (32, 64):
        raise ValueError('Supported input sizes: 32, 64')
    return int(math.log2(256 // input_size))
