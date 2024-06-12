# image_encoder_inference.py

import torch
from torchvision import transforms
from PIL import Image
import os

class ImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)
    
    def forward(self, x):
        return self.resnet(x)

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def inference(image_dir, model_path, output_dim=512):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = ImageEncoder(output_dim=output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            image = load_image(image_path, transform).to(device)
            features = model(image)
            print(f'Features for {image_name}: {features.cpu().numpy()}')

if __name__ == '__main__':
    image_dir = 'path/to/your/test/image/folder'
    model_path = 'image_encoder.pth'
    inference(image_dir, model_path)

