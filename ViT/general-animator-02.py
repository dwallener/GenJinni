import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, num_classes, dim, depth, heads, mlp_dim):
        super(VisionTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim

        self.to_patches = nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patches(img)
        x = x.flatten(2).transpose(1, 2)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x).view(b, 3, self.image_size, self.image_size)    

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage


class ImageDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.images = [f for f in os.listdir(img_path) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def interpolate_images(model, images, num_frames):
    model.eval()
    interpolated = []
    with torch.no_grad():
        for i in range(len(images)):
            start_img = images[i].unsqueeze(0)
            end_img = images[(i + 1) % len(images)].unsqueeze(0)
            for j in range(num_frames):
                alpha = (j + 1) / (num_frames + 1)
                blended_img = (1 - alpha) * start_img + alpha * end_img
                interpolated_img = model(blended_img)
                interpolated.append(interpolated_img.squeeze(0))
    return interpolated


def save_images(images, epoch, save_path):
    to_pil = ToPILImage()
    for i, img in enumerate(images):
        img = to_pil(img.cpu())
        img.save(os.path.join(save_path, f'interpolated_epoch_{epoch}_img_{i}.png'))


image_size = 32
patch_size = 4
num_channels = 3
num_classes = image_size * image_size * 3
dim = 512
depth = 6
heads = 8
mlp_dim = 1024
num_epochs = 100
num_frames = 2
img_path = '../../artwork/animation_collision_training_imgs'
save_path = 'interpolated_images'

model = VisionTransformer(image_size, patch_size, num_channels, num_classes, dim, depth, heads, mlp_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
transform = transforms.Compose([transforms.Resize((image_size, image_size)), ToTensor()])
dataset = ImageDataset(img_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        interpolated = interpolate_images(model, batch, num_frames)
        batch = batch.unsqueeze(0).repeat(num_frames, 1, 1, 1, 1).view(-1, 3, image_size, image_size)
        loss = criterion(torch.stack(interpolated), batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}')
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            interpolated = interpolate_images(model, batch, num_frames)
            save_images(interpolated, epoch, save_path)