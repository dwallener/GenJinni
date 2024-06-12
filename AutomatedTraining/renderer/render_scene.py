# render_scene.py

from panda3d.core import Point3, Texture, GraphicsOutput
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
import random
import os
from PIL import Image
import numpy as np

import json

class Panda3DRenderer(ShowBase):
    def __init__(self, output_dir):
        ShowBase.__init__(self)
        self.output_dir = output_dir
        self.movements = []
        self.setup_scene()
        self.camera.set_pos(0, 0, 0)
        self.camera.look_at(0, 1, 0)
        self.accept("escape", self.exit_program)
        self.accept("a", self.move_camera, ["A"])
        self.accept("d", self.move_camera, ["D"])
        self.accept("w", self.move_camera, ["W"])
        self.accept("s", self.move_camera, ["S"])
        self.accept("q", self.rotate_camera, ["Q"])
        self.accept("e", self.rotate_camera, ["E"])
        self.taskMgr.add(self.update, "update")

    def move_camera(self, direction):
        self.movements.append(direction)
        if direction == "A":
            self.camera.set_x(self.camera.get_x() - 0.1)
        elif direction == "D":
            self.camera.set_x(self.camera.get_x() + 0.1)
        elif direction == "W":
            self.camera.set_y(self.camera.get_y() + 0.1)
        elif direction == "S":
            self.camera.set_y(self.camera.get_y() - 0.1)

    def rotate_camera(self, direction):
        self.movements.append(direction)
        if direction == "Q":
            self.camera.set_h(self.camera.get_h() + 5)
        elif direction == "E":
            self.camera.set_h(self.camera.get_h() - 5)

    def update(self, task):
        self.capture_frame()
        return Task.cont

    def capture_frame(self):
        image = self.win.get_screenshot()
        image_data = np.array(image)
        image_pil = Image.fromarray(image_data)
        frame_path = os.path.join(self.output_dir, f"{task.frame}.jpg")
        image_pil.save(frame_path)

    def exit_program(self):
        self.save_movements()
        self.userExit()
        exit()

    def save_movements(self):
        movements_path = os.path.join(self.output_dir, "movements.json")
        with open(movements_path, "w") as f:
            json.dump(self.movements, f)


if __name__ == "__main__":
    output_dir = "output_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    app = Panda3DRenderer(output_dir)
    app.run()
