#!/usr/bin/python3

# collision classifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image


class CollisionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
    

class CustomTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes, patch_size=16, num_patches=196):
        super(CustomTransformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.embedding = nn.Linear(patch_size * patch_size * 3, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = x.view(batch_size, self.num_patches, -1)
        x = self.embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding

        x = self.transformer(x)
        cls_output = x[:, 0]
        logits = self.fc(cls_output)
        return logits


def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')


def test_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    loss = running_loss / len(dataloader.dataset)
    accuracy = 100 * correct / total
    print(f'Test Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your dataset
    train_image_paths = [...]  # List of training image paths
    train_labels = [...]  # List of training labels
    test_image_paths = [...]  # List of test image paths
    test_labels = [...]  # List of test labels

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create datasets and dataloaders
    train_dataset = CollisionDataset(train_image_paths, train_labels, transform)
    test_dataset = CollisionDataset(test_image_paths, test_labels, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, criterion, and optimizer
    embed_dim = 256
    num_heads = 8
    num_layers = 6
    num_classes = 4  # 4 classes: no collision, obstacle 1, obstacle 2, obstacle 3
    model = CustomTransformer(embed_dim, num_heads, num_layers, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train_model(model, train_dataloader, criterion, optimizer, num_epochs=10)

    # Test the model
    test_model(model, test_dataloader, criterion)
    



