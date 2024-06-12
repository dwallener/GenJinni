# train_image_encoder.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

class ImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=False)  # Start with untrained model
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)
    
    def forward(self, x):
        return self.resnet(x)

def train_image_encoder(data_dir, output_dim=512, batch_size=32, num_epochs=20, learning_rate=1e-4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ImageEncoder(output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(data_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    torch.save(model.state_dict(), 'image_encoder.pth')
    print('Model saved as image_encoder.pth')

if __name__ == '__main__':
    data_dir = 'path/to/your/image/folder'
    train_image_encoder(data_dir)

