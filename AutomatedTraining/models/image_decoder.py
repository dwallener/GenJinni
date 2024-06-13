# image_decoder.py

import torch.nn as nn

class ImageDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ImageDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, kernel_size=4, stride=1),  # 1x1 -> 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # 128x128 -> 256x256
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x)