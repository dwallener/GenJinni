#!/usr/local/bin/python3

# Generate a simulated simulated result

import os
import numpy as np
import random
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

path_to_orig = "output/panda_frame_caps"

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


# checkpoints
checkpoint_dir = '/home/damir00/Sandbox/transformer-from-scratch/checkpoints/panda_frame_caps/'
checkpoint_path = '/home/damir00/Sandbox/transformer-from-scratch/checkpoints/panda_frame_caps/e0300_main_checkpoint.pth'
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


model = load_model(checkpoint_path)
initial_img_path = 'dataset/training/panda_frame_caps/a_encoded/frame-encoded-00047.png'
initial_img = Image.open(initial_img_path).convert('RGB')
# Transform to tensor
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])
input_img = transform(initial_img).unsqueeze(0)  # Add batch dimension

# Generate 1000 frames
output_dir = 'output/panda_frame_caps/frames'

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
