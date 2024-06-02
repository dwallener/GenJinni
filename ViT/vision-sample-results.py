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


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.n_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, emb_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)  # (batch_size, emb_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, emb_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, emb_dim)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads

        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.fc_out = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        batch_size, n_patches, emb_dim = x.shape

        q = self.query(x).reshape(batch_size, n_patches, self.num_heads, self.head_dim)
        k = self.key(x).reshape(batch_size, n_patches, self.num_heads, self.head_dim)
        v = self.value(x).reshape(batch_size, n_patches, self.num_heads, self.head_dim)

        qk = torch.einsum("bqhd,bkhd->bhqk", q, k) / (self.head_dim ** 0.5)
        attention = torch.softmax(qk, dim=-1)

        out = torch.einsum("bhqk,bkhd->bqhd", attention, v).reshape(batch_size, n_patches, emb_dim)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, forward_expansion * emb_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_dim, emb_dim)
        )

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, num_heads, num_layers, forward_expansion, num_classes, img_size):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.n_patches, emb_dim))

        self.layers = nn.ModuleList(
            [
                TransformerBlock(emb_dim, num_heads, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position_embeddings

        for layer in self.layers:
            x = layer(x)

        cls_token_final = x[:, 0]
        out = self.fc_out(cls_token_final)
        return out

# setup
device = torch.device('cpu')

# sample sequence length
num_images = 10

# checkpoints
checkpoint_dir = 'checkpoints/panda_frame_caps'
dataset_dir = 'dataset/training/panda_frame_caps'

# TODO: Import these from the same class as used by training
# Hyperparameters and paths
img_size = 128
patch_size = 8
in_channels = 3
emb_dim = 1024
num_heads = 4
num_layers = 18
forward_expansion = 4
num_classes = img_size * img_size * in_channels  # Assuming next frame prediction with same size
batch_size = 4


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


def display_and_save_plot(source_images, target_images, output_images, epoch, folder):
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
    #plt.savefig(f'output/multi-track-samples/main-epoch-{epoch:04d}-{folder}.png')
    plt.savefig(f'output/panda_frame_caps/main-epoch-{epoch:04d}-{folder}.png')
    
    # plt.show()


def main():
    # checkpoints = sorted(os.listdir(checkpoint_dir))
    # warmup OR main
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if 'main' in f])
    print(checkpoints)

    for epoch, checkpoint_file in enumerate(checkpoints):
        epoch_source_images = []
        epoch_target_images = []
        epoch_output_images = []

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        model = load_model(checkpoint_path)

        root_dir = 'dataset/training/panda_frame_caps/'
        items = os.listdir(root_dir)

        folders = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]


        # an N-frame sequence starting at a random point, from each training set
        for folder in folders:
            
            if folder == 'a_naked':
                source_dir = root_dir + folder
                # Read images
                print(f"Reading images...{source_dir}")
                M = len(os.listdir(source_dir))
                # start with lowest possible frame index
                P = random.randint(30, M-N-1)
                # for the smaller runs of set_length = 500
                # P = random.randint(30, 450)
                print(f"{folder}: P: {P} Pend: {P+N} M: {M}")
                source_indices = list(range(P, P+N))
                target_indices = list(range(P+1, P+N+1))

                prefix = 'frame-'
                if folder == 'hk-patches': prefix = 'patch_'
                if folder == 'mk64-rr-source': prefix = ''
                if folder == 'racer-track': prefix = 'racer-track-'
                if folder == 'sv-patches': prefix = 'patch_'


                source_images = read_images(root_dir+folder, source_indices, prefix)
                target_images = read_images(root_dir+folder, target_indices, prefix)

                # run the model
                print("Running model...")
                with torch.no_grad():
                    output_images = model(source_images)
                
                # dump the results
                display_and_save_plot(source_images, target_images, output_images, epoch, folder)

        # run the model
        print("Running model...")
        with torch.no_grad():
            output_images = model(source_images)

        # dump the results
        display_and_save_plot(source_images, target_images, output_images, epoch, folder)

if __name__ == "__main__":
    N = 10
    main()

