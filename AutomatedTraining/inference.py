# inference.py

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models import ImageEncoder, TextEncoder, TransformerDecoder, ImageDecoder
import os

"""

1.	Inference Class:
	•	Initialization: Loads the trained model and sets it to evaluation mode.
	•	Process Image: Preprocesses the input image to match the format expected by the model.
	•	Predict Next Frame: Takes the current image and a command, encodes them using the trained models, and generates the predicted next frame.
	•	Command to Index: Maps movement commands to their corresponding indices.
	•	Save Image: Converts the predicted frame from a tensor to an image and saves it to disk.
2.	Main Execution:
	•	Loads the trained model from model.pth.
	•	Loads the current frame from current_image_path.
	•	Uses a sample command (‘W’ in this example).
	•	Predicts the next frame based on the current frame and command.
	•	Saves the predicted frame to output_image_path.
"""

class Inference:
    def __init__(self, model_path, output_dim=512, vocab_size=6, embed_dim=256, hidden_dim=512, n_layers=6, n_heads=8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_encoder = ImageEncoder(output_dim=output_dim).to(self.device)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads).to(self.device)
        self.transformer_decoder = TransformerDecoder(embed_dim=embed_dim, n_heads=n_heads, hidden_dim=hidden_dim, n_layers=n_layers).to(self.device)
        self.image_decoder = ImageDecoder(input_dim=hidden_dim).to(self.device)

        self.load_model(model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.image_encoder.load_state_dict(checkpoint['image_encoder'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.transformer_decoder.load_state_dict(checkpoint['transformer_decoder'])
        self.image_decoder.load_state_dict(checkpoint['image_decoder'])

        self.image_encoder.eval()
        self.text_encoder.eval()
        self.transformer_decoder.eval()
        self.image_decoder.eval()

    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image.to(self.device)

    def predict_next_frame(self, current_image, command):
        with torch.no_grad():
            current_image = self.process_image(current_image)
            command_tensor = torch.tensor([self.command_to_index(command)]).unsqueeze(0).to(self.device)  # Add batch dimension

            image_features = self.image_encoder(current_image)
            text_features = self.text_encoder(command_tensor)
            combined_features = self.transformer_decoder(text_features, image_features)
            predicted_image = self.image_decoder(combined_features)

            return predicted_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert back to image format

    def command_to_index(self, command):
        command_map = {'W': 0, 'A': 1, 'S': 2, 'D': 3, 'Q': 4, 'E': 5}
        return command_map[command]

    def save_image(self, image_array, output_path):
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        image.save(output_path)

if __name__ == "__main__":
    model_path = 'model.pth'
    current_image_path = 'path/to/current/image.jpg'
    command = 'W'  # Example command
    output_image_path = 'path/to/save/predicted/image.jpg'

    inference = Inference(model_path)
    next_frame = inference.predict_next_frame(current_image_path, command)
    inference.save_image(next_frame, output_image_path)

