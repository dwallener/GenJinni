# render_scene.py

from panda3d.core import Point3, Texture, GraphicsOutput
from panda3d.core import Point3, WindowProperties

from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.interval.IntervalGlobal import Sequence

import random
import os
from PIL import Image
import numpy as np

class Panda3DRenderer(ShowBase):
    def __init__(self, output_dir):
        ShowBase.__init__(self)

        # Set window properties
        props = WindowProperties()
        props.setSize(128, 128)
        self.win.requestProperties(props)

        # Disable the camera trackball controls.
        self.disableMouse()

        # Load the environment model.
        self.scene = self.loader.loadModel("models/environment")
        
        # Calculate the current dimensions of the scene.
        min_point, max_point = self.scene.getTightBounds()
        current_dimensions = max_point - min_point
        print(f"Current scene dimensions: {current_dimensions}")

        # Reparent the model to render and apply position transforms.
        self.scene.reparentTo(self.render)
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        # Add the spinCameraTask procedure to the task manager.
        # self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # Load and transform the panda actor.
        self.pandaActor = Actor("models/panda-model",
                                {"walk": "models/panda-walk4"})
        self.pandaActor.setScale(0.005, 0.005, 0.005)
        self.pandaActor.reparentTo(self.render)

        # Loop its animation.
        self.pandaActor.loop("walk")

        # Create the four lerp intervals needed for the panda to
        # walk back and forth.
        posInterval1 = self.pandaActor.posInterval(13,
                                                   Point3(0, -10, 0),
                                                   startPos=Point3(0, 10, 0))
        posInterval2 = self.pandaActor.posInterval(13,
                                                   Point3(0, 10, 0),
                                                   startPos=Point3(0, -10, 0))
        hprInterval1 = self.pandaActor.hprInterval(3,
                                                   Point3(180, 0, 0),
                                                   startHpr=Point3(0, 0, 0))
        hprInterval2 = self.pandaActor.hprInterval(3,
                                                   Point3(0, 0, 0),
                                                   startHpr=Point3(180, 0, 0))

        # Create and play the sequence that coordinates the intervals.
        self.pandaPace = Sequence(posInterval1, hprInterval1,
                                  posInterval2, hprInterval2,
                                  name="pandaPace")
        self.pandaPace.loop()

        # Initialize camera position and rotation variables.
        self.angle_horizontal = 0
        self.angle_vertical = 0
        self.radius = 20
        self.height = 3

        self.output_dir = output_dir
        self.frame_count = 0
        self.setup_scene()
        self.camera.set_pos(0, 0, 0)
        self.camera.look_at(0, 1, 0)
        self.accept("escape", self.exit_program)
        self.taskMgr.add(self.update, "update")

    def setup_scene(self):
        # self.scene = self.loader.load_model("3d-world-01.egg")
        # self.scene.reparent_to(self.render)
        self.set_background_color(0, 0, 0, 1)
        self.disableMouse()

    def move_camera(self, direction):
        if direction == "A":
            self.camera.set_x(self.camera.get_x() - 0.1)
        elif direction == "D":
            self.camera.set_x(self.camera.get_x() + 0.1)
        elif direction == "W":
            self.camera.set_y(self.camera.get_y() + 0.1)
        elif direction == "S":
            self.camera.set_y(self.camera.get_y() - 0.1)
        elif direction == "Q":
            self.camera.set_h(self.camera.get_h() + 5)
        elif direction == "E":
            self.camera.set_h(self.camera.get_h() - 5)

    def update(self, task):
        return Task.cont

    def capture_frame(self, frame_path):
        image = self.win.get_screenshot()
        image_data = np.array(image)
        image_pil = Image.fromarray(image_data)
        image_pil.save(frame_path)

    def exit_program(self):
        self.userExit()
        exit()

if __name__ == "__main__":
    output_dir = "output_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    app = Panda3DRenderer(output_dir)
    app.run()