#!/usr/local/bin/python3

# Play it like a game

import os
import numpy as np
import random
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from VisionTransformer import VisionTransformer

def load_model(checkpoint_path):
    model = VisionTransformer(
        in_channels=in_channels,
        patch_size=patch_size,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        forward_expansion=forward_expansion,
        num_classes=num_classes,
        img_size=img_size   
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


model = load_model(f'{checkpoint_dir}/e0300_main_checkpoint.pth')

from Hyperparameters import Hyperparameters
hp = Hyperparameters()

img_size = hp.img_size
patch_size = hp.patch_size
in_channels = hp.in_channels
emb_dim = hp.emb_dim
num_heads = hp.num_heads
num_layers = hp.num_layers
forward_expansion = hp.forward_expansion
num_classes = hp.num_classes
batch_size = hp.batch_size

# checkpoints
checkpoint_dir = 'panda3d'

model = load_model(f'{checkpoint_dir}/e0300_main_checkpoint.pth')

initial_img_path = '../panda3d/frame_caps/naked/frame-00010.png'
initial_img = Image.open(initial_img_path).convert('RGB')
# Transform to tensor
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])
input_img = transform(initial_img).unsqueeze(0)  # Add batch dimension

# now let's do a gameplay loop...

# start with initial image
# read WASD
# apply encoding
# generate output

# Generate 1000 frames
output_dir = 'output/panda_sim'

model.eval()
for i in range(1000):
    with torch.no_grad():
        output_img = model(input_img).cpu().view(in_channels, img_size, img_size)

    # Save the output image
    output_img_pil = transforms.ToPILImage()(output_img)
    output_img_path = os.path.join(output_dir, f'oframe_{i:05d}.png')
    output_img_pil.save(output_img_path)

    # Use the output as the new input
    input_img = output_img.unsqueeze(0)
    if i % 10 == 0:
        print(f"Index: {i:05d}")

print("Generated 1000 frames.")

# Generate an MP4 from the resulting output images
frame_rate = 30  # Define the frame rate
output_video_path = 'output/panda_frame_caps.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (img_size, img_size))

for i in range(1000):
    frame_path = os.path.join(output_dir, f'oframe_{i:05d}.png')
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

video_writer.release()
print(f"MP4 video saved to {output_video_path}.")
