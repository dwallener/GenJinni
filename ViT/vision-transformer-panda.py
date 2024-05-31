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
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png'))])        
        self.transform = transform

    def __len__(self):
        # The length is one less than the number of images, as the last image has no subsequent image
        return len(self.images) - 2

    def __getitem__(self, idx):
        source_img_path = os.path.join(self.image_dir, self.images[idx])
        target_img_path = os.path.join(self.image_dir, self.images[idx + 1])

        source_image = Image.open(source_img_path).convert('RGB')
        target_image = Image.open(target_img_path).convert('RGB')

        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        target_image = target_image.view(-1)  # Flatten target image

        return source_image, target_image

# the original one, for using separate dirs
class PairedImageDataset2(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_images = sorted(os.listdir(source_dir))
        self.target_images = sorted(os.listdir(target_dir))
        self.transform = transform

    def __len__(self):
        return len(self.source_images) - 1

    def __getitem__(self, idx):
        source_img_path = os.path.join(self.source_dir, self.source_images[idx])
        target_img_path = os.path.join(self.target_dir, self.target_images[idx+1])

        source_image = Image.open(source_img_path).convert('RGB')
        target_image = Image.open(target_img_path).convert('RGB')

        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        target_image = target_image.view(-1)  # Flatten target image

        return source_image, target_image


# TODO: Put this in a class and import it so training and running can use the same values
# Hyperparameters and paths
img_size = 64
patch_size = 4
in_channels = 3
emb_dim = 1024
num_heads = 4
num_layers = 24
forward_expansion = 4
num_classes = img_size * img_size * in_channels  # Assuming next frame prediction with same size
batch_size = 4

# training length control
# warmup_epochs = 50
# main_epochs = 500

training_sets = ['hk-patches', 'mk64-rr-source', 'sv-patches', 'racer-track']
set_length = 1200

# TODO: This section is outdated...clean up or toss
source_dir = f'dataset/training/{root_path}'
target_dir = f'dataset/training/{root_path}'
#

# Transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
print(f"Arrangine the dataset...")
# dataset = PairedImageDataset(source_dir, target_dir, transform)
dataset = PairedImageDataset(source_dir, transform)
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
checkpoint_dir = f'checkpoints/{root_path}'
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
remaining_warmup_epochs = max(0, max_warmup_epochs - start_epoch_warmup)
remaining_main_epochs = max(0, max_epochs - start_epoch_main)

# Training loop

# TODO: Need to clean this up to better reflect 'root_name'
#training_dir = 'dataset/training'
#subdirs = [os.path.join(training_dir, d) for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))]
training_dir = 'dataset/training/panda_frame_caps'
subdirs = ['dataset/training/panda_frame_caps']

# TODO: Need to allow two separate dirs for source/target, to cover the encoded use case

# Warmup phase
print(f"Warmup training phase...")
# check for existing warmup checkpoint
for epoch in range(start_epoch_warmup, start_epoch_warmup + remaining_warmup_epochs):

    # every epoch we pull a random frame and store the model results
    #test_frame = random.randint(0,500)

    # move on if we've already done this
    if epoch >= max_warmup_epochs:
        break
    model.train()

    sample_frame = 0

    for subdir in subdirs:
        print(f"Sourcing from training set {subdir}")
        source_subdir = subdir + '/a_encoded'
        target_subdir = subdir + '/a_naked'
        #dataset = PairedImageDataset(subdir, transform=transform)
        dataset = PairedImageDataset2(source_subdir, target_subdir, transform=transform)
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

# TODO: clean this mess up...everything needs to derive from root_name
training_dir = 'dataset/training'
subdirs = [os.path.join(training_dir, d) for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))]
training_dir = 'dataset/training/panda_frame_caps'
subdirs = ['dataset/training/panda_frame_caps']


# Main training phase
for epoch in range(start_epoch_main, start_epoch_main + remaining_main_epochs):
    # move on if we've already done this
    if epoch >= max_epochs:
        break
    model.train()
    for subdir in subdirs:
        print(f"Sourcing from training set {subdir}")
        #dataset = PairedImageDataset(subdir, transform=transform)
        source_subdir = subdir + '/a_encoded'
        target_subdir = subdir + '/a_naked'
        dataset = PairedImageDataset2(source_subdir, target_subdir, transform=transform)
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

# now generate a complete run...
print("Generating a complete run...")

import cv2

# Directories
output_dir = 'output/racer-track/'
os.makedirs(output_dir, exist_ok=True)

# Load the first image from the dataset
source_dir = 'dataset/training/mk-rr-source/'
initial_img_path = os.path.join(source_dir, sorted(os.listdir(source_dir))[30])
initial_img = Image.open(initial_img_path).convert('RGB')

# Transform to tensor
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])
input_img = transform(initial_img).unsqueeze(0)  # Add batch dimension

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_img = input_img.to(device)

# Generate 1000 frames
model.eval()
for i in range(1000):
    with torch.no_grad():
        output_img = model(input_img).cpu().view(in_channels, img_size, img_size)

    # Save the output image
    output_img_pil = transforms.ToPILImage()(output_img)
    output_img_path = os.path.join(output_dir, f'frame_{i:04d}.png')
    output_img_pil.save(output_img_path)

    # Use the output as the new input
    input_img = output_img.unsqueeze(0).to(device)

print("Generated 1000 frames.")

# Generate an MP4 from the resulting output images
frame_rate = 30  # Define the frame rate
output_video_path = 'output/racer-track-generated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (img_size, img_size))

for i in range(1000):
    frame_path = os.path.join(output_dir, f'frame_{i:04d}.png')
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

video_writer.release()
print(f"MP4 video saved to {output_video_path}.")
