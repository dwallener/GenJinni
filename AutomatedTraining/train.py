# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from models import ImageEncoder, TextEncoder, TransformerDecoder, ImageDecoder
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

# Transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_encoder = ImageEncoder(output_dim=image_dim).to(device)
text_encoder = TextEncoder(vocab_size=text_vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads).to(device)
transformer_decoder = TransformerDecoder(embed_dim=embed_dim, n_heads=n_heads, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
image_decoder = ImageDecoder(input_dim=hidden_dim).to(device)

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
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
renderer = Panda3DRenderer(output_dir)

# Function to capture and process image
def capture_and_process_image(renderer, frame_count):
    frame_path = os.path.join(output_dir, f"{frame_count}.jpg")
    renderer.capture_frame(frame_path)
    image = Image.open(frame_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device), frame_path

# Training Loop
frame_count = 0
current_image, current_image_path = capture_and_process_image(renderer, frame_count)

for epoch in range(num_epochs):
    for _ in range(batch_size):
        optimizer.zero_grad()
        
        movement = camera_control.get_next_movement()
        renderer.move_camera(movement)
        frame_count += 1
        next_image, next_image_path = capture_and_process_image(renderer, frame_count)
        
        # Convert movement to tensor
        movement_tensor = torch.tensor([camera_control.command_to_index(movement)]).unsqueeze(0).to(device)

        # Forward pass
        image_features = image_encoder(current_image)
        text_features = text_encoder(movement_tensor)
        combined_features = transformer_decoder(text_features, image_features)
        predicted_image = image_decoder(combined_features)

        # Compute loss
        loss = criterion(predicted_image, next_image)
        loss.backward()
        optimizer.step()

        # Update current image for next iteration
        current_image = next_image

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save model
torch.save({
    'image_encoder': image_encoder.state_dict(),
    'text_encoder': text_encoder.state_dict(),
    'transformer_decoder': transformer_decoder.state_dict(),
    'image_decoder': image_decoder.state_dict()
}, 'model.pth')