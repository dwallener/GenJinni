# render_scene.py

from panda3d.core import Point3, WindowProperties, DirectionalLight, AmbientLight
from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.interval.IntervalGlobal import Sequence
import os


class Panda3DRenderer(ShowBase):
    def __init__(self, output_dir):
        ShowBase.__init__(self)

        # Set window properties
        props = WindowProperties()
        props.setSize(224, 224)
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

        # Add lighting
        self.add_lighting()

        # Load and transform the panda actor.
        self.pandaActor = Actor("models/panda-model", {"walk": "models/panda-walk4"})
        self.pandaActor.setScale(0.005, 0.005, 0.005)
        self.pandaActor.reparentTo(self.render)

        # Loop its animation.
        self.pandaActor.loop("walk")
        self.pandaActor.stop()  # Stop the animation initially

        # Create the four lerp intervals needed for the panda to walk back and forth.
        posInterval1 = self.pandaActor.posInterval(13, Point3(0, -10, 0), startPos=Point3(0, 10, 0))
        posInterval2 = self.pandaActor.posInterval(13, Point3(0, 10, 0), startPos=Point3(0, -10, 0))
        hprInterval1 = self.pandaActor.hprInterval(3, Point3(180, 0, 0), startHpr=Point3(0, 0, 0))
        hprInterval2 = self.pandaActor.hprInterval(3, Point3(0, 0, 0), startHpr=Point3(180, 0, 0))

        # Create and play the sequence that coordinates the intervals.
        self.pandaPace = Sequence(posInterval1, hprInterval1, posInterval2, hprInterval2, name="pandaPace")
        self.pandaPace.loop()
        self.pandaPace.pause()  # Pause the sequence initially

        # Initialize camera position and rotation variables.
        self.angle_horizontal = 0
        self.angle_vertical = 0
        self.radius = 20
        self.height = 3

        self.output_dir = output_dir
        self.frame_count = 0
        self.setup_scene()
        self.camera.set_pos(0, -50, 10)  # Set the camera position to view the scene
        self.camera.look_at(self.scene)  # Make the camera look at the center of the scene
        self.accept("escape", self.exit_program)
        self.accept("a", self.move_camera, ["A"])
        self.accept("d", self.move_camera, ["D"])
        self.accept("w", self.move_camera, ["W"])
        self.accept("s", self.move_camera, ["S"])
        self.accept("q", self.move_camera, ["Q"])
        self.accept("e", self.move_camera, ["E"])
        self.taskMgr.add(self.update, "update")

    def add_lighting(self):
        # Add a directional light
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(Point3(0, 0, -1))
        directionalLight.setColor((1, 1, 1, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        self.render.setLight(directionalLightNP)

        # Add an ambient light
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((0.2, 0.2, 0.2, 1))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)

    def setup_scene(self):
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
        
        self.advance_panda_animation()

    def advance_panda_animation(self):
        # Manually advance the panda animation by one frame
        current_frame = self.pandaActor.getCurrentFrame('walk')
        next_frame = (current_frame + 1) % self.pandaActor.getNumFrames('walk')
        self.pandaActor.pose('walk', next_frame)

    def update(self, task):
        return Task.cont

    def capture_frame(self, frame_path):
        self.win.saveScreenshot(frame_path)

    def exit_program(self):
        self.userExit()
        exit()


def run_renderer(output_dir):
    app = Panda3DRenderer(output_dir)
    # app.run()
    return app


def run_renderer_with_task(output_dir, task_func):
    app = Panda3DRenderer(output_dir)
    app.taskMgr.add(task_func, "training_task")
    app.run()
    return app


if __name__ == "__main__":
    output_dir = "output_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    run_renderer(output_dir)

