#1/usr/bin/python3

# collision animation interpolator

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

# control input/output size, assumes square
image_size = 224
num_interpolations = 10


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_path = '../../artwork/animation_collision_training_imgs/sprite-sheet-frame-'
image_paths = [f'{img_path}00.png', f'{img_path}01.png', f'{img_path}02.png', f'{img_path}03.png']
transform = get_transform(image_size)
dataset = ImageDataset(image_paths, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


class VisionTransformer(nn.Module):
    def __init__(self, image_size=image_size, num_interpolations=num_interpolations):
        super(VisionTransformer, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.num_interpolations = num_interpolations
        self.image_size = image_size
        self.decoder = nn.Sequential(
            nn.Linear(1000, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size * image_size * 3),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size, _, _, _ = x.size()
        encoded = self.vit(x)
        interpolated_images = self.decoder(encoded)
        interpolated_images = interpolated_images.view(batch_size * self.num_interpolations, 3, image_size, image_size)
        return interpolated_images


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(num_interpolations=num_interpolations).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# if necessary, create the dir to save the interpolated output
save_dir = 'interpolated_images'
os.makedirs(save_dir, exist_ok=True)

epochs = 10
for epoch in range(epochs):
    for i, images in enumerate(dataloader):
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images.repeat(num_interpolations, 1, 1, 1))
        loss.backward()
        optimizer.step()

        # dump the output
        #TODO: eventually change this to happen only after epoch or multiple epochs

        for j, img in enumerate(outputs):
            img = img.cpu().detach().numpy().transpose(1, 2, 0)  # Convert tensor to numpy array and rearrange dimensions
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)  # Normalize to [0, 255]
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(save_dir, f'interpolated_epoch{epoch+1}_batch{i+1}_img{j+1}.png'))

        # dump epoch training deets
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

