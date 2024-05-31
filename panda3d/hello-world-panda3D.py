from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3, WindowProperties
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import TransparencyAttrib

class MyApp(ShowBase):
    def __init__(self):
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
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

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

        # Set up key event handling.
        self.accept('a', self.set_key, ['left', True])
        self.accept('d', self.set_key, ['right', True])
        self.accept('w', self.set_key, ['up', True])
        self.accept('s', self.set_key, ['down', True])
        self.accept('A', self.set_key, ['left', True])
        self.accept('D', self.set_key, ['right', True])
        self.accept('W', self.set_key, ['up', True])
        self.accept('S', self.set_key, ['down', True])
        
        self.accept('a-up', self.set_key, ['left', False])
        self.accept('d-up', self.set_key, ['right', False])
        self.accept('w-up', self.set_key, ['up', False])
        self.accept('s-up', self.set_key, ['down', False])
        self.accept('A-up', self.set_key, ['left', False])
        self.accept('D-up', self.set_key, ['right', False])
        self.accept('W-up', self.set_key, ['up', False])
        self.accept('S-up', self.set_key, ['down', False])

        # Track key state
        self.keys = {'left': False, 'right': False, 'up': False, 'down': False}

        # Initialize overlay attribute
        self.square_overlay = None

        # Initial direction.
        self.rotation_direction = None
        self.frame_counter = 0

    def set_key(self, key, value):
        self.keys[key] = value
        self.rotation_direction = key if value else None

    def add_square_overlay(self, direction):
        positions = {
            'up': (0, 0, 0.8),
            'down': (0, 0, -0.8),
            'left': (-0.8, 0, 0),
            'right': (0.8, 0, 0)
        }
        pos = positions[direction]
        square = OnscreenImage(image='red_square.png', pos=pos, scale=0.04, parent=render2d)
        square.setTransparency(TransparencyAttrib.MAlpha)
        return square

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        dt = globalClock.getDt() * 0.5

        # Update the camera angle based on the direction.
        if self.keys['left']:
            self.angle_horizontal += dt
        elif self.keys['right']:
            self.angle_horizontal -= dt
        elif self.keys['up']:
            self.angle_vertical += dt
        elif self.keys['down']:
            self.angle_vertical -= dt

        # Limit vertical rotation to prevent going below ground level.
        self.angle_vertical = max(-pi/2, min(pi/2, self.angle_vertical))

        # Ensure the camera doesn't go below ground level.
        x = self.radius * cos(self.angle_horizontal)
        y = self.radius * sin(self.angle_horizontal)
        z = max(self.height, self.height + self.radius * sin(self.angle_vertical))

        # Update camera position and look-at point.
        self.camera.setPos(Point3(x, y, z))
        self.camera.lookAt(0, 0, 0)

        if self.rotation_direction:
            # Render the red square indicating the input direction
            if self.square_overlay:
                self.square_overlay.destroy()
            self.square_overlay = self.add_square_overlay(self.rotation_direction)

            # Save the current frame as a PNG file, with and without encoding
            self.square_overlay.hide()
            self.win.saveScreenshot(f"frame_caps/frame-{self.frame_counter:05d}.png")
            self.square_overlay.show()
            self.win.saveScreenshot(f"frame_caps/frame-encoded-{self.frame_counter:05d}.png")

            self.frame_counter += 1
        elif self.square_overlay:
            self.square_overlay.destroy()
            self.square_overlay = None

        return Task.cont

app = MyApp()
app.run()