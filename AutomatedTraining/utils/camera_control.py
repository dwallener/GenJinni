# camera_control.py

import random


class CameraControl:
    def __init__(self, method="random"):
        self.method = method

    def get_next_movement(self):
        if self.method == "random":
            return self.random_movement()
        else:
            return self.algorithmic_movement()

    def random_movement(self):
        return random.choice(["A", "D", "W", "S", "Q", "E"])

    def algorithmic_movement(self):
        # Stub for future algorithmic movement
        return self.random_movement()
    
