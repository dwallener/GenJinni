# data_loader.py

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class NavigationDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.frames = sorted([f for f in os.listdir(data_dir) if f.endswith(".jpg")])
        self.movements = self.load_movements()

    def load_movements(self):
        with open(os.path.join(self.data_dir, "movements.json"), "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.frames) - 1

    def __getitem__(self, idx):
        current_frame = Image.open(os.path.join(self.data_dir, self.frames[idx])).convert("RGB")
        next_frame = Image.open(os.path.join(self.data_dir, self.frames[idx + 1])).convert("RGB")
        movement = self.movements[idx]

        current_frame = torch.tensor(np.array(current_frame)).permute(2, 0, 1) / 255.0
        next_frame = torch.tensor(np.array(next_frame)).permute(2, 0, 1) / 255.0
        movement = torch.tensor(movement)

        return current_frame, movement, next_frame


def save_movement(data_dir, frame_idx, movement):
    movements_path = os.path.join(data_dir, "movements.json")
    if os.path.exists(movements_path):
        with open(movements_path, "r") as f:
            movements = json.load(f)
    else:
        movements = []

    movements.append(movement)
    with open(movements_path, "w") as f:
        json.dump(movements, f)

