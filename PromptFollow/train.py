# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from models import ImageEncoder, TextEncoder, TransformerDecoder, ImageDecoder
from utils.data_loader import get_data_loader

# Hyperparameters
image_dim = 512
text_vocab_size = 1000
embed_dim = 256
hidden_dim = 512
n_heads = 8
n_layers = 6
batch_size = 32
learning_rate = 1e-4
num_epochs = 20

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# DataLoader
data_loader = get_data_loader('data/images', 'data/commands.txt', transform, batch_size)

# Model
image_encoder = ImageEncoder(output_dim=image_dim)
text_encoder = TextEncoder(vocab_size=text_vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads)
transformer_decoder = TransformerDecoder(embed_dim=embed_dim, n_heads=n_heads, hidden_dim=hidden_dim, n_layers=n_layers)
image_decoder = ImageDecoder(input_dim=hidden_dim)

# Optimizer and Loss
optimizer = optim.Adam(
    list(image_encoder.parameters()) + 
    list(text_encoder.parameters()) + 
    list(transformer_decoder.parameters()) + 
    list(image_decoder.parameters()), 
    lr=learning_rate
)
criterion = nn.MSELoss()

# Training Loop
for epoch in range(num_epochs):
    for images, commands in data_loader:
        optimizer.zero_grad()
        
        image_features = image_encoder(images)
        text_features = text_encoder(commands)
        combined_features = transformer_decoder(text_features, image_features)
        predicted_images = image_decoder(combined_features)

        loss = criterion(predicted_images, images)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save model
torch.save({
    'image_encoder': image_encoder.state_dict(),
    'text_encoder': text_encoder.state_dict(),
    'transformer_decoder': transformer_decoder.state_dict(),
    'image_decoder': image_decoder.state_dict()
}, 'model.pth')

