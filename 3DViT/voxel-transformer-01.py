#!/usr/bin/python3

# first experiment with voxel transformer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Voxel Creation

import numpy as np

def create_time_series(length, voxel_size, initial_position, voxel_size_obj):
    """
    Create a time series of 3D voxel grids with a 2x2 object moving upwards and downwards.

    :param length: Total number of time steps.
    :param voxel_size: Size of the 3D grid (NxNxN).
    :param initial_position: Initial position of the moving voxel (bottom center).
    :param voxel_size_obj: Size of the moving voxel object (2x2x2).
    :return: A numpy array of shape (length, 1, voxel_size, voxel_size, voxel_size) representing the time series.
    """
    # Initialize the 3D time series with all zeros
    time_series = np.zeros((length, 1, voxel_size, voxel_size, voxel_size), dtype=np.float32)
    
    # Define the initial position
    current_position = list(initial_position)
    
    # Define the gravity and velocity parameters
    velocity = 1.0
    gravity = -0.5
    ascending_steps = 3
    
    for t in range(length):
        # Clear previous position
        if t > 0:
            x, y, z = previous_position
            time_series[t, 0, x:x+voxel_size_obj, y:y+voxel_size_obj, z:z+voxel_size_obj] = 0
        
        # Set new position
        x, y, z = current_position
        time_series[t, 0, x:x+voxel_size_obj, y:y+voxel_size_obj, z:z+voxel_size_obj] = 1
        
        # Update position for the next time step
        previous_position = list(current_position)
        if t < ascending_steps:
            current_position[0] -= int(velocity)  # Move up
            velocity += gravity
        else:
            current_position[0] += int(velocity)  # Move down
            velocity += gravity
        
        # Clamp the position within the grid boundaries
        current_position[0] = max(0, min(current_position[0], voxel_size - voxel_size_obj))
        current_position[1] = max(0, min(current_position[1], voxel_size - voxel_size_obj))
        current_position[2] = max(0, min(current_position[2], voxel_size - voxel_size_obj))
        
    return time_series


# visualize the time series
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

def animate_voxels(time_series):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def update_plot(frame):
        ax.clear()
        ax.set_xlim(0, time_series.shape[2])
        ax.set_ylim(0, time_series.shape[3])
        ax.set_zlim(0, time_series.shape[4])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        data = time_series[frame, 0]
        x, y, z = np.where(data > 0)

        ax.scatter(x, y, z, zdir='z', c='red')

        ax.set_title(f'Time Step: {frame}')

    ani = animation.FuncAnimation(fig, update_plot, frames=len(time_series), interval=200)
    plt.show()

# Parameters for the time series
length = 20
voxel_size = 32
initial_position = (30, 15, 15)
voxel_size_obj = 2

# Create the time series
time_series = create_time_series(length, voxel_size, initial_position, voxel_size_obj)

# Animate the time series
animate_voxels(time_series)

# Parameters
length = 20
voxel_size = 32
initial_position = (30, 15, 15)
voxel_size_obj = 2

# Create the time series
time_series = create_time_series(length, voxel_size, initial_position, voxel_size_obj)

# Verify the output
for t in range(length):
    print(f"Time step {t}:")
    print(time_series[t, 0, :, :, :])

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
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

print("Training complete.")

