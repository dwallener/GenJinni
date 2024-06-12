# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from models import ImageEncoder, TextEncoder, TransformerDecoder, ImageDecoder
from utils.data_loader import NavigationDataset, save_movement
from utils.camera_control import CameraControl
from renderer.render_scene import Panda3DRenderer

# Hyperparameters
image_dim = 512
text_vocab_size = 6  # For WASDQE
embed_dim = 256
hidden_dim = 512
n_heads = 8
n_layers = 6
batch_size = 32
learning_rate = 1e-4
num_epochs = 20
output_dir = "output_frames"

# Dataset and DataLoader
dataset = NavigationDataset(output_dir)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
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

# Camera control
camera_control = CameraControl(method="random")  # Change to "algorithm" as needed

# Initialize Panda3D
renderer = Panda3DRenderer(output_dir)
renderer.taskMgr.add(renderer.update, "update")

# Training Loop
for epoch in range(num_epochs):
    for images, commands, targets in data_loader:
        optimizer.zero_grad()
        
        image_features = image_encoder(images)
        text_features = text_encoder(commands)
        combined_features = transformer_decoder(text_features, image_features)
        predicted_images = image_decoder(combined_features)

        loss = criterion(predicted_images, targets)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Move camera and capture next frame
    movement = camera_control.get_next_movement()
    renderer.move_camera(movement)
    renderer.capture_frame()

# Save model
torch.save({
    'image_encoder': image_encoder.state_dict(),
    'text_encoder': text_encoder.state_dict(),
    'transformer_decoder': transformer_decoder.state_dict(),
    'image_decoder': image_decoder.state_dict()
}, 'model.pth')

