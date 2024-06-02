#!/usr/bin/python3

# first experiment with voxel transformer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class VoxelDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx + 1]

class PatchEmbedding3D(nn.Module):
    def __init__(self, voxel_size, patch_size, emb_dim):
        super(PatchEmbedding3D, self).__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.linear = nn.Linear(patch_size**3, emb_dim)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size**3)
        x = self.linear(x)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, -1, self.emb_dim)
        return x

class VisionTransformer3D(nn.Module):
    def __init__(self, voxel_size, patch_size, emb_dim, depth, num_heads, mlp_dim):
        super(VisionTransformer3D, self).__init__()
        self.patch_embedding = PatchEmbedding3D(voxel_size, patch_size, emb_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, (voxel_size // patch_size)**3, emb_dim))
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=mlp_dim)
            for _ in range(depth)
        ])
        self.output_layer = nn.Linear(emb_dim, patch_size**3)
        self.patch_size = patch_size
        self.voxel_size = voxel_size
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x += self.positional_encoding
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.view(-1, 1, self.voxel_size, self.voxel_size, self.voxel_size)
        return x

# Hyperparameters
voxel_size = 32
patch_size = 4
emb_dim = 128
depth = 6
num_heads = 8
mlp_dim = 256
batch_size = 8
epochs = 10
learning_rate = 0.001

# Generate some dummy data
M = 100
data = np.random.randn(M, 1, voxel_size, voxel_size, voxel_size).astype(np.float32)

# Create DataLoader
dataset = VoxelDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = VisionTransformer3D(voxel_size, patch_size, emb_dim, depth, num_heads, mlp_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = torch.tensor(inputs), torch.tensor(targets)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

print("Training complete.")

