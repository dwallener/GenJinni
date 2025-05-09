#!//usr/local/bin/python3

# based on ViT-02, making it specific for dealing with Panda3D
# basically, the 3D sim generates two folders...
# naked/ which holds raw screen grabs, on a stutter of N
# overlay/ which holds the red-square overlay indicating direction

# based on ViT-01, adding support for multiple image stream
# because apparently we need crap tons of training images


# handle input params
import argparse

parser = argparse.ArgumentParser(description='Train GenJinn Transformer')
parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs')
parser.add_argument('--max_warmup_epochs', type=int, default=10, help='Number of low learning rate epochs to start with')
parser.add_argument('--root_name', type=str, required=True, help='name for subfolder under checkpoints and dataset/training')
args = parser.parse_args()

max_epochs = args.max_epochs
max_warmup_epochs = args.max_warmup_epochs
root_path = args.root_name

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import random

import platform


def get_os_type():
    os_type = platform.system()
    if os_type == "Linux":
        return "linux"
    elif os_type == "Darwin":
        return "macos"
    else:
        return "Unknown"


# set up CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == torch.device('cuda'):
    print("CUDA!!!!")
else:
    print("Booo!!!")

device = torch.device('cuda')


# pretty print preamble
def config_pretty_print(root_path, num_params, heads, layers, batch_size, img_size, patch_size, epochs_warmup, epochs_main, lr_warmup, lr_main):
    print(f'Starting a run...')
    print(f'Pulling images from dataset/training/{root_path}')
    print(f'')
    print(f'Storing checkpoints in checkpoints/{root_path}')
    print(f'')
    print(f'Parameter count: {num_params}')
    print(f'Heads: {heads} Layers: {layers} Batch Size: {batch_size}')
    print(f'')
    print(f'Image size: {img_size}')
    print(f'Patch size: {patch_size}')
    print(f'')
    print(f'Warmup Epochs   : {epochs_warmup}')
    print(f'Training Epochs : {epochs_main}')
    print(f'')
    print(f'Warmup LR: {lr_warmup}')
    print(f'Main LR  : {lr_main}')
    print(f'')
    print(f'')


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

import os
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class PairedImageDataset(Dataset):
    # here image_dir is the root folder for naked/overlay
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.naked_dir = f'{image_dir}/naked/'
        self.overlay_dir = f'{image_dir}/overlay/'
        self.naked_images = sorted([f for f in os.listdir(self.naked_dir) if f.lower().endswith(('.png'))])
        self.overlay_images = sorted([f for f in os.listdir(self.overlay_dir) if f.lower().endswith(('.png'))])
        print(f'Found {len(self.naked_images)} naked images...')        
        print(f'Found {len(self.overlay_images)} overlay images...')        
        self.transform = transform
        print(f"Sourcing images from {self.naked_dir}")
        print(f"Sourcing encoding from {self.overlay_dir}")

    def __len__(self):
        # The length is one less than the number of images, as the last image has no subsequent image
        return len(self.naked_images) - 2

    def __getitem__(self, idx):
        naked_img_path = os.path.join(self.naked_dir, self.naked_images[idx])
        target_img_path = os.path.join(self.naked_dir, self.naked_images[idx + 1])
        overlay_img_path = os.path.join(self.overlay_dir, self.overlay_images[idx])

        # RGBA...?
        naked_image = Image.open(naked_img_path).convert('RGBA')
        overlay_image = Image.open(overlay_img_path).convert('RGBA')
        source_image = Image.alpha_composite(naked_image, overlay_image).convert('RGB')
        target_image = Image.open(target_img_path).convert('RGB')

        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        target_image = target_image.view(-1)  # Flatten target image

        return source_image, target_image


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

# training length control
# warmup_epochs = 50
# main_epochs = 500

set_length = 5000

# Transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

os_type = get_os_type()
if os_type == 'macos':
    # macos
    training_dir = '/Users/damir00/Sandbox/GenJinni/panda3d/frame_caps'
else:
    # ubuntu
    training_dir = '/home/damir00/Sandbox/GenJinni/panda3d/frame_caps'

# Dataset and DataLoader
print(f"Arrangine the dataset...")
# we sort out the overlay, target, source frames inside the called function
print(f'Construcing dataset...')
dataset = PairedImageDataset(training_dir, transform)
print('Constructing dataloader...')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
print(f"Instantiating the vision tranformer model...")
print(f"")
model = VisionTransformer(
    in_channels=in_channels,
    patch_size=patch_size,
    emb_dim=emb_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    forward_expansion=forward_expansion,
    num_classes=num_classes,
    img_size=img_size
).to(device)

# Dump transformer deets
num_parameters = sum(p.numel() for p in model.parameters())
print(f"Model Parameters: {num_parameters}")
print(f"")

# Loss and optimizer
criterion = nn.MSELoss()
initial_lr = 1e-8  # 100x lower than the main learning rate
main_lr = 1e-6

# Set up training (normal fp32)
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
# set up training (fp16/bf16)
scaler = GradScaler()

# Dump training run deets in preamble
config_pretty_print(root_path, num_parameters, num_heads, num_layers, batch_size, img_size, patch_size, max_warmup_epochs, max_epochs, initial_lr, main_lr)

# Manage checkpoints

# Checkpoint paths
checkpoint_dir = f'panda3d'
os.makedirs(checkpoint_dir, exist_ok=True)
warmup_checkpoint_path = os.path.join(checkpoint_dir, 'warmup_checkpoint.pth')
main_checkpoint_path = os.path.join(checkpoint_dir, 'main_checkpoint.pth')

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, scaler, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict' : scaler.state_dict(),
    }, checkpoint_path)


# Function to load checkpoint
def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    if os.path.exists(checkpoint_path):
        print(f'Found a checkpoint file...')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    else:
        print(f'No checkpoint file...')
        return 0


# Load checkpoint if it exists
start_epoch_warmup = load_checkpoint(warmup_checkpoint_path, model, optimizer, scaler)
start_epoch_main = load_checkpoint(main_checkpoint_path, model, optimizer, scaler)

# calculate remaining epochs
rem_warm_epochs = max(0, max_warmup_epochs - start_epoch_warmup)
remaining_main_epochs = max(0, max_epochs - start_epoch_main)


# Training loop

# Warmup phase
print(f"Warmup training phase...")
# check for existing warmup checkpoint
for epoch in range(start_epoch_warmup, start_epoch_warmup + rem_warm_epochs):

    # move on if we've already done this
    if epoch >= max_warmup_epochs:
        break
    model.train()

    # what does this do...?
    # sample_frame = 0

    # source and target are created inside this next call
    # including compositing the naked and encoded images
    dataset = PairedImageDataset(training_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (source, target) in enumerate(dataloader):

        if batch_idx > set_length / batch_size:
            break

        source, target = source.to(device), target.to(device)
        # Forward pass
        with autocast():
            outputs = model(source).to(device)
            loss = criterion(outputs, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        # this is the fp32 line
        # loss.backward()
        # optimizer.step()
        # this is the fp16 lines
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (batch_idx + 1) % 10 == 0:
            print(f'Warmup Epoch [{epoch+1}/{max_warmup_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            print(f'Warmup Epoch [{epoch+1}/{max_warmup_epochs}], Step [{batch_idx*batch_size}/{set_length}], Loss: {loss.item():.4f}')
                

    # Save checkpoint
    if epoch % 10 == 0:
        warmup_checkpoint_path = os.path.join(checkpoint_dir, f'e{epoch:04d}_warmup_checkpoint.pth')
        save_checkpoint(model, optimizer, epoch+1, scaler, warmup_checkpoint_path)


print(f"Main training phase...")

# Update optimizer for main training phase
for param_group in optimizer.param_groups:
    param_group['lr'] = main_lr

# be able to source from multiple training datasets 
# organized as 'dataset/training/...'

# reuse the same training dir...
training_dir = training_dir

# Main training phase
for epoch in range(start_epoch_main, start_epoch_main + remaining_main_epochs):
    # move on if we've already done this
    if epoch >= max_epochs:
        break
    model.train()

    print(f"Sourcing from training set {training_dir}")

    dataset = PairedImageDataset(training_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (source, target) in enumerate(dataloader):

        if batch_idx > set_length / batch_size:
            break

        source, target = source.to(device), target.to(device)
        # Forward pass
        with autocast():
            outputs = model(source)
            loss = criterion(outputs, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        # fp32 way
        # loss.backward()
        # optimizer.step()
        # this is the auto fp16 way
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{max_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch+1}/{max_epochs}], Step [{batch_idx*batch_size}/{set_length}], Loss: {loss.item():.4f}')                

        if batch_idx > set_length:
            break

    # turn this on to save checkpoint on each epoch
    # torch.save(model.state_dict(), f'checkpoints/mk64-rr-checkpoint-epoch{epoch+1}.pth')
    # Checkpoint the latest epoch
    # Save checkpoint
    if epoch % 100 == 0:
        main_checkpoint_path = os.path.join(checkpoint_dir, f'e{epoch:04d}_main_checkpoint.pth')
        save_checkpoint(model, optimizer, epoch+1, scaler, main_checkpoint_path)


print("Training completed")

# Now let's evaluate...

import random
import matplotlib.pyplot as plt


# Function to display and save the images
def display_and_save_composition(source_img, target_img, output_img, filename='result.png'):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(source_img.permute(1, 2, 0))
    axes[0].set_title("Source Image")
    axes[0].axis('off')

    axes[1].imshow(target_img.permute(1, 2, 0))
    axes[1].set_title("Target Image")
    axes[1].axis('off')

    axes[2].imshow(output_img.permute(1, 2, 0))
    axes[2].set_title("Output Image")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# Select a random source image from the dataset
random_index = random.randint(0, len(dataset) - 1)
source_img, target_img = dataset[random_index]
source_img = source_img.unsqueeze(0)  # Add batch dimension

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
source_img = source_img.to(device)

# Generate the output image
model.eval()
with torch.no_grad():
    output_img = model(source_img).cpu().view(in_channels, img_size, img_size)

# Convert the images back to their original shape
source_img = source_img.squeeze(0).cpu()
target_img = target_img.view(in_channels, img_size, img_size).cpu()

# Display and save the images
display_and_save_composition(source_img, target_img, output_img, filename='result.png')
