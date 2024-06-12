# image_decoder.py

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ImageDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ImageDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x)
    

