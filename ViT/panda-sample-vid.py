#!/usr/local/bin/python3

import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from VisionTransformer import VisionTransformer

# setup
device = torch.device('cpu')

# sample sequence length
num_images = 10

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


def read_images(image_dir, indices, prefix):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    images = []
    for idx in indices:
        img_path = os.path.join(image_dir, f'{prefix}{idx:05d}.png')
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)
    return torch.stack(images)


import cv2
import numpy as np


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.
    Assumes the tensor is in the format (C, H, W) and values are in the range [0, 1].
    """
    tensor = tensor.detach().cpu()  # Detach from graph and move to CPU
    tensor = tensor.permute(1, 2, 0).numpy()  # Rearrange dimensions to (H, W, C)
    tensor = (tensor * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    return tensor


def generate_movie(source_tensors, target_tensors, output_tensors, output_filename='panda-sample-video.mp4'):

    source_imgs = [tensor_to_image(tensor) for tensor in source_tensors]
    target_imgs = [tensor_to_image(tensor) for tensor in target_tensors]
    output_imgs = [tensor_to_image(tensor) for tensor in output_tensors]
    
    # Ensure all lists have the same length
    assert len(source_imgs) == len(target_imgs) == len(output_imgs), "All image lists must have the same length"

    # Get the dimensions of the images
    height, width, layers = source_imgs[0].shape
    
    # Create a red border
    border_color = (0, 0, 255)  # Red in BGR format
    border_thickness = 2  # Thickness of the border
    
    # Calculate the dimensions of the output video
    bordered_height = height + 2 * border_thickness
    bordered_width = width + 2 * border_thickness
    video_width = 3 * bordered_width  # Three panels horizontally
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # Frames per second
    video = cv2.VideoWriter(output_filename, fourcc, fps, (video_width, bordered_height))
    
    for src_img, tgt_img, out_img in zip(source_imgs, target_imgs, output_imgs):
        # Add a red border to each image
        src_img_bordered = cv2.copyMakeBorder(src_img, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)
        tgt_img_bordered = cv2.copyMakeBorder(tgt_img, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)
        out_img_bordered = cv2.copyMakeBorder(out_img, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)
        
        # Concatenate the images horizontally
        frame = np.concatenate((src_img_bordered, tgt_img_bordered, out_img_bordered), axis=1)
        
        # Write the frame to the video
        video.write(frame)
    
    # Release the video writer
    video.release()
    print(f"Video saved as {output_filename}")


def main():

    # checkpoints
    checkpoint_dir = 'panda3d'
    dataset_dir = '../panda3d/frame_caps/naked'

    checkpoint_path = os.path.join(checkpoint_dir, 'e1500_main_checkpoint.pth')
    model = load_model(checkpoint_path)

    # Read images
    print(f"Reading images...{dataset_dir}")
    # upper bound
    M = len(os.listdir(dataset_dir))
    # lower bound
    N = 10 
    # list length
    P = 200
    # stutter length
    Q = 5 # set to whatever the frame cap stutter is
    rounded_random_int = round(random.randint(N, M-P) / Q) * Q
    source_indices = [rounded_random_int + Q * i for i in range(P)]
    target_indices = [value + Q for value in source_indices]

    prefix = 'frame-'
    source_images = read_images(dataset_dir, source_indices, prefix)
    target_images = read_images(dataset_dir, target_indices, prefix)

    # run the model
    print("Running model...")
    with torch.no_grad():
        output_images = model(source_images)
    
    # dump the results
    generate_movie(source_images, target_images, output_images)

if __name__ == "__main__":
    main()

