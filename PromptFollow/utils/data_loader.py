# data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class NavigationDataset(Dataset):
    def __init__(self, image_dir, command_file, transform=None):
        self.image_dir = image_dir
        self.commands = open(command_file).read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        command, image_name = self.commands[idx].split(',')
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, command

def get_data_loader(image_dir, command_file, transform, batch_size, shuffle=True):
    dataset = NavigationDataset(image_dir, command_file, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

