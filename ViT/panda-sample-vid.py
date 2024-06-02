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


def display_and_save_plot(source_images, target_images, output_images, epoch):
    fix, axes = plt.subplots(10, 3, figsize=(15, 30))
    for i in range(10):
        axes[i, 0].imshow(transforms.ToPILImage()(source_images[i]))
        axes[i,0].set_title('Source')
        axes[i, 1].imshow(transforms.ToPILImage()(target_images[i]))
        axes[i,1].set_title('Target')
        axes[i, 2].imshow(transforms.ToPILImage()(output_images[i].view(in_channels, img_size, img_size)))
        axes[i,2].set_title('Output')
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'panda3d/mepoch-{epoch:05d}.png')
    plt.close()


def save_movie(source_images, target_images, output_images, epoch, frames):

    fix, axes = plt.subplots(frames, 3, figsize=(15, 30))
    for i in range(frames):
        axes[i, 0].imshow(transforms.ToPILImage()(source_images[i]))
        axes[i,0].set_title('Source')
        axes[i, 1].imshow(transforms.ToPILImage()(target_images[i]))
        axes[i,1].set_title('Target')
        axes[i, 2].imshow(transforms.ToPILImage()(output_images[i].view(in_channels, img_size, img_size)))
        axes[i,2].set_title('Output')
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'panda3d/mepoch-{epoch:05d}.png')
    plt.close()


def save_movie()
    pass

def main():

    # checkpoints
    checkpoint_dir = 'panda3d'
    dataset_dir = '../panda3d/frame_caps/naked'

    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if 'main' in f])
    print(checkpoints)

    for epoch, checkpoint_file in enumerate(checkpoints):

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        model = load_model(checkpoint_path)

        items = os.listdir(dataset_dir)

        # Read images
        print(f"Reading images...{dataset_dir}")
        # upper bound
        M = len(os.listdir(dataset_dir))
        # lower bound
        N = 10 
        # list length
        P = 10 
        # stutter length
        Q = 5
        rounded_random_int = round(random.randint(N, M) / Q) * Q
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
        save_movie(source_images, target_images, output_images, epoch, P)


if __name__ == "__main__":
    N = 10
    main()

