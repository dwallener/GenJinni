import pygame
import numpy as np
import random
import math
from enum import Enum

# Direction enum and PixelObject class definition

class Direction(Enum):
    N = 0
    NNE = 1
    NE = 2
    ENE = 3
    E = 4
    ESE = 5
    SE = 6
    SSE = 7
    S = 8
    SSW = 9
    SW = 10
    WSW = 11
    W = 12
    WNW = 13
    NW = 14
    NNW = 15

    @classmethod
    def from_value(cls, value):
        """Get the direction from a value (0-15)."""
        for direction in cls:
            if direction.value == value:
                return direction
        raise ValueError("Value must be within the range 0-15")

class PixelObject:
    DIRECTION_ANGLES = {
        Direction.N: 270,
        Direction.NNE: 292.5,
        Direction.NE: 315,
        Direction.ENE: 337.5,
        Direction.E: 0,
        Direction.ESE: 22.5,
        Direction.SE: 45,
        Direction.SSE: 67.5,
        Direction.S: 90,
        Direction.SSW: 112.5,
        Direction.SW: 135,
        Direction.WSW: 157.5,
        Direction.W: 180,
        Direction.WNW: 202.5,
        Direction.NW: 225,
        Direction.NNW: 247.5
    }

    def __init__(self, R=None, G=None, B=None, dest_x=64, dest_y=64):
        self.destination = (dest_x, dest_y)

        if R is None or G is None:
            # Randomly position along one of the edges of the 128x128 plane
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                self.R = np.uint8(random.randint(0, 127))
                self.G = np.uint8(0)
            elif edge == 'bottom':
                self.R = np.uint8(random.randint(0, 127))
                self.G = np.uint8(127)
            elif edge == 'left':
                self.R = np.uint8(0)
                self.G = np.uint8(random.randint(0, 127))
            elif edge == 'right':
                self.R = np.uint8(127)
                self.G = np.uint8(random.randint(0, 127))
        else:
            self.R = np.uint8(R)
            self.G = np.uint8(G)

        if B is None:
            initial_speed = 1
            self.B = np.uint8(initial_speed)
        else:
            self.B = np.uint8(B)

        self.set_direction_towards_destination()
        self.set_safety_buffer()

    def set_safety_buffer(self):
        """Calculate and set the safety buffer to 1/2 the distance between the target destination and the edge of the plane."""
        distance_to_left_edge = self.destination[0]
        distance_to_right_edge = 127 - self.destination[0]
        distance_to_top_edge = self.destination[1]
        distance_to_bottom_edge = 127 - self.destination[1]

        nearest_edge_distance = min(distance_to_left_edge, distance_to_right_edge, distance_to_top_edge, distance_to_bottom_edge)
        self.safety_buffer = nearest_edge_distance / 2

    def set_direction_towards_destination(self):
        """Set the direction towards the destination coordinates."""
        delta_x = self.destination[0] - self.R
        delta_y = self.destination[1] - self.G

        angle = math.degrees(math.atan2(delta_y, delta_x))
        if angle < 0:
            angle += 360

        closest_direction = min(self.DIRECTION_ANGLES.keys(), key=lambda d: abs(self.DIRECTION_ANGLES[d] - angle))
        self.set_direction(closest_direction)

    def update_R(self, new_R):
        """Update the horizontal position."""
        if 0 <= new_R <= 127:
            self.R = np.uint8(new_R)
        else:
            raise ValueError("R must be within the range 0-127")
    
    def update_G(self, new_G):
        """Update the vertical position."""
        if 0 <= new_G <= 127:
            self.G = np.uint8(new_G)
        else:
            raise ValueError("G must be within the range 0-127")
    
    def update_B(self, new_B):
        """Update the direction and speed."""
        if 0 <= new_B <= 255:
            self.B = np.uint8(new_B)
        else:
            raise ValueError("B must be within the range 0-255")
    
    def get_direction(self):
        """Get the direction from the upper 4 bits of B."""
        direction_value = (self.B >> 4) & 0x0F
        return Direction.from_value(direction_value)
    
    def get_speed(self):
        """Get the speed from the lower 4 bits of B."""
        return self.B & 0x0F
    
    def set_direction(self, direction):
        """Set the direction in the upper 4 bits of B."""
        if isinstance(direction, Direction):
            direction_value = direction.value
            self.B = (self.B & 0x0F) | (direction_value << 4)
        else:
            raise ValueError("Direction must be an instance of the Direction enum")
    
    def set_speed(self, speed):
        """Set the speed in the lower 4 bits of B."""
        if 0 <= speed <= 15:
            self.B = (self.B & 0xF0) | speed
        else:
            raise ValueError("Speed must be within the range 0-15")

    def update_position(self, all_objects):
        """Update the position based on the current speed and direction, respecting the safety buffer."""
        if not hasattr(self, 'vibrating'):
            self.vibrating = False
            self.original_position = (self.R, self.G)

        # Calculate the direction towards the destination
        self.set_direction_towards_destination()
        direction_angle = self.DIRECTION_ANGLES[self.get_direction()]
        speed = self.get_speed()

        delta_x = int(round(speed * math.cos(math.radians(direction_angle))))
        delta_y = int(round(speed * math.sin(math.radians(direction_angle))))

        new_R = np.clip(self.R + delta_x, 0, 127)
        new_G = np.clip(self.G + delta_y, 0, 127)

        # Check safety buffer
        new_distance_to_destination = math.sqrt((self.destination[0] - new_R) ** 2 + (self.destination[1] - new_G) ** 2)

        # Convert self.R and self.G to int for the neighbor count calculation
        int_R = int(self.R)
        int_G = int(self.G)

        # Count other PixelObjects in the 3x3 area centered on this PixelObject
        neighbor_count = sum(1 for obj in all_objects if obj is not self and abs(int(obj.R) - int_R) <= 1 and abs(int(obj.G) - int_G) <= 1)

        # Decrease the safety buffer if there are more than 5 neighbors
        if neighbor_count > 5:
            self.safety_buffer = max(0, self.safety_buffer - 1)

        if new_distance_to_destination < self.safety_buffer:
            self.vibrating = True

        if self.vibrating:
            if (self.R, self.G) == self.original_position:
                # Move 1 pixel in a random direction
                random_direction = random.choice([Direction.N, Direction.NE, Direction.E, Direction.SE, Direction.S, Direction.SW, Direction.W, Direction.NW])
                direction_angle = self.DIRECTION_ANGLES[random_direction]
                delta_x = int(round(math.cos(math.radians(direction_angle))))
                delta_y = int(round(math.sin(math.radians(direction_angle))))
                new_R = np.clip(self.R + delta_x, 0, 127)
                new_G = np.clip(self.G + delta_y, 0, 127)
                self.R, self.G = new_R, new_G
            else:
                # Return to the original stopping point
                self.R, self.G = self.original_position
        else:
            self.R = np.uint8(new_R)
            self.G = np.uint8(new_G)
            if new_distance_to_destination >= self.safety_buffer:
                self.original_position = (self.R, self.G)  # Update the original position when moving normally

    def __repr__(self):
        return f"PixelObject(R={self.R}, G={self.G}, B={self.B}, Direction={self.get_direction().name}, Speed={self.get_speed()}, Destination={self.destination}, SafetyBuffer={self.safety_buffer})"

# Initialize Pygame and set up the display
pygame.init()
screen = pygame.display.set_mode((512, 512))  # Scale the display by 4x
clock = pygame.time.Clock()
running = True

# List to store PixelObjects
pixel_objects = []

# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            # Create a new PixelObject with a destination at the center (64, 64)
            pixel = PixelObject(dest_x=64, dest_y=64)
            pixel_objects.append(pixel)

            # Update positions of existing PixelObjects
            for obj in pixel_objects:
                obj.update_position(pixel_objects)

            # Clear the screen
            screen.fill((0, 0, 0))

            # Mark the location of each PixelObject in red
            for obj in pixel_objects:
                screen.fill((255, 0, 0), (obj.R * 4, obj.G * 4, 4, 4))

            # Mark the destination in green
            screen.fill((0, 255, 0), (64 * 4, 64 * 4, 4, 4))

            # Update the display
            pygame.display.flip()

    # Cap the frame rate
    clock.tick(30)

pygame.quit()