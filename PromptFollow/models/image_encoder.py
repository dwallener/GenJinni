# image_encoder.py

import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)
    
    def forward(self, x):
        return self.resnet(x)
    
