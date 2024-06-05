#!/usr/bin/python3

# generate a gameplay sequence


# Load background (static for this demo)
# Load initial actor image
# Load initial npc images
# Place npc on background

from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def load_and_prepare_image(image, scale_factor=1.0):
    # Load the image
    #image = Image.open(filepath)
    
    # Convert white background to transparent
    image = image.convert("RGBA")
    datas = image.getdata()
    
    new_data = []
    for item in datas:
        # Change all white (also shades of whites)
        # to transparent
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    image.putdata(new_data)
    
    # Scale the image
    if scale_factor != 1.0:
        image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)), Image.LANCZOS)
    
    return image


def create_scene():
    # Load the background image and resize it to 1080p (1920x1080)
    background = Image.open('../../artwork/animation_collision_training_imgs/arena-sprite-01.png').resize((1920, 1080))

    # Load and prepare the actor and NPC images
    actor = load_and_prepare_image(Image.open('../../artwork/animation_collision_training_imgs/actor-sprite-00.png'), scale_factor=0.3)
    npc_02 = load_and_prepare_image('../../artwork/animation_collision_training_imgs/npc-sprite-02.png', scale_factor=0.3)
    npc_00 = load_and_prepare_image('../../artwork/animation_collision_training_imgs/npc-sprite-00.png', scale_factor=0.3)
    npc_01 = load_and_prepare_image('../../artwork/animation_collision_training_imgs/npc-sprite-01.png', scale_factor=0.3)

    # Calculate positions
    actor_position = (int(1920 * 0.1), int(1080 * 0.5 - actor.size[1] / 2) + 20)  # Left edge, halfway down
    npc_02_position = (int(1920 * 0.9 - npc_02.size[0]), int(1080 * 0.5 - npc_02.size[1] / 2))  # Right edge, halfway down
    npc_00_position = (int(actor_position[0] + (npc_02_position[0] - actor_position[0]) / 3), int(1080 * 0.5 - npc_00.size[1] / 2))  # 1/3 between actor and npc_02
    npc_01_position = (int(actor_position[0] + 2 * (npc_02_position[0] - actor_position[0]) / 3), int(1080 * 0.5 - npc_01.size[1] / 2))  # 2/3 between actor and npc_02

    # Paste the actor and NPC images onto the background
    background.paste(actor, actor_position, actor)
    background.paste(npc_02, npc_02_position, npc_02)
    background.paste(npc_00, npc_00_position, npc_00)
    background.paste(npc_01, npc_01_position, npc_01)

    # Save the final image
    background.save('final_scene.png')

    # Display the final image
    plt.imshow(background)
    plt.axis('off')  # Hide the axes
    plt.show()

# load up the generative animations

import cv2


def read_video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames


# Run the function to create the scene
# create_scene()

# Loop...

# Move npc on WASD (only AD are active here) (update animation frame + position)
# Test for collisions
# If collision == true:
#     get collision classification (could be any of the npc)
#     select next actor frame
#     select next npc frame
#     if end of animation sequence:
#         break

import pygame
import cv2
import numpy as np
from PIL import Image

# Initialize pygame
pygame.init()

# Example placeholders for global variables
actor_position = [100, 540]  # Starting position for the actor
actor_sprites = [
    load_and_prepare_image(Image.open('../../artwork/animation_collision_training_imgs/actor-sprite-00.png'), scale_factor=0.3),
    load_and_prepare_image(Image.open('../../artwork/animation_collision_training_imgs/actor-sprite-01.png'), scale_factor=0.3),
    load_and_prepare_image(Image.open('../../artwork/animation_collision_training_imgs/actor-sprite-02.png'), scale_factor=0.3)
]
actor_anim = read_video_to_frames('../../artwork/animation_collision_training_imgs/actor-anim.mp4')
npc_anim = [read_video_to_frames('../../artwork/animation_collision_training_imgs/npc-anim.mp4'), read_video_to_frames('../../artwork/animation_collision_training_imgs/npc-anim.mp4'), read_video_to_frames('../../artwork/animation_collision_training_imgs/npc-anim.mp4')]
npc_positions = [
    [640, 540],  # Position for npc-00
    [960, 540],  # Position for npc-01
    [1280, 540]  # Position for npc-02
]

background = Image.open('../../artwork/animation_collision_training_imgs/arena-sprite.png').resize((1920, 1080))
screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("Actor and NPC Animation")


def load_and_prepare_image(image, scale_factor=1.0):
    # Convert white background to transparent
    image = image.convert("RGBA")
    datas = image.getdata()
    
    new_data = []
    for item in datas:
        # Change all white (also shades of whites)
        # to transparent
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    image.putdata(new_data)
    
    # Scale the image
    if scale_factor != 1.0:
        image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)), Image.LANCZOS)
    
    return image


def rect_overlap(rect1, rect2):
    # Check if two rectangles overlap
    return not (rect1[0] > rect2[0] + rect2[2] or
                rect1[0] + rect1[2] < rect2[0] or
                rect1[1] > rect2[1] + rect2[3] or
                rect1[1] + rect1[3] < rect2[1])


def collision_check():
    actor_rect = (actor_position[0], actor_position[1], actor_sprites[0].width, actor_sprites[0].height)
    
    for i, npc_pos in enumerate(npc_positions):
        npc_rect = (npc_pos[0], npc_pos[1], npc_anim[i][0].shape[1], npc_anim[i][0].shape[0])
        if rect_overlap(actor_rect, npc_rect):
            return [1 if i == j else 0 for j in range(3)]
    
    return [0, 0, 0]


def play_frames(frames, actor_pos=None, npc_pos=None):
    for frame in frames:
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        frame_pil = load_and_prepare_image(frame_pil, scale_factor=0.3)
        updated_scene = background.copy()
        
        if actor_pos:
            updated_scene.paste(frame_pil, actor_pos, frame_pil)
        if npc_pos:
            updated_scene.paste(frame_pil, npc_pos, frame_pil)
        
        updated_scene_np = np.array(updated_scene)
        updated_scene_cv2 = cv2.cvtColor(updated_scene_np, cv2.COLOR_RGB2BGR)
        updated_scene_pygame = pygame.image.frombuffer(updated_scene_cv2.tobytes(), updated_scene_cv2.shape[1::-1], "RGB")
        
        screen.blit(updated_scene_pygame, (0, 0))
        pygame.display.flip()
        pygame.time.delay(33)


def play_frames_simul(actor_frames, npc_frames, actor_pos, npc_pos):
    for actor_frame, npc_frame in zip(actor_frames, npc_frames):
        actor_frame_pil = Image.fromarray(cv2.cvtColor(actor_frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        npc_frame_pil = Image.fromarray(cv2.cvtColor(npc_frame, cv2.COLOR_BGR2RGB)).convert("RGBA")

        actor_frame_pil = load_and_prepare_image(actor_frame_pil, scale_factor=0.3)
        npc_frame_pil = load_and_prepare_image(npc_frame_pil, scale_factor=0.3)

        updated_scene = background.copy()
        updated_scene.paste(actor_frame_pil, actor_pos, actor_frame_pil)
        updated_scene.paste(npc_frame_pil, npc_pos, npc_frame_pil)

        updated_scene_np = np.array(updated_scene)
        updated_scene_cv2 = cv2.cvtColor(updated_scene_np, cv2.COLOR_RGB2BGR)
        updated_scene_pygame = pygame.image.frombuffer(updated_scene_cv2.tobytes(), updated_scene_cv2.shape[1::-1], "RGB")

        screen.blit(updated_scene_pygame, (0, 0))
        pygame.display.flip()
        pygame.time.delay(33)


def update():
    global actor_position
    clock = pygame.time.Clock()
    actor_frame_index = 0
    actor_direction = 'right'

    position_delta = 12

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        moving = False
        if keys[pygame.K_a]:
            if actor_position[0] > 100:  # Prevent moving left beyond starting position
                actor_position[0] -= position_delta
                actor_direction = 'left'
                moving = True
                collision_result = collision_check()
                for i in range(3):
                    if collision_result[i] != 0:
                        #play_frames(actor_anim, actor_pos=actor_position)
                        #play_frames(npc_anim[i], npc_pos=npc_positions[i])
                        play_frames_simul(actor_anim, npc_anim[i], actor_position, npc_positions[i])
        if keys[pygame.K_d]:
            actor_position[0] += position_delta
            actor_direction = 'right'
            moving = True
            collision_result = collision_check()
            for i in range(3):
                if collision_result[i] != 0:
                    #play_frames(actor_anim, actor_pos=actor_position)
                    #play_frames(npc_anim[i], npc_pos=npc_positions[i])
                    play_frames_simul(actor_anim, npc_anim[i], actor_position, npc_positions[i])
        
        # Update the scene with the current positions
        updated_scene = background.copy()
        # Update the scene with the current positions
        updated_scene = background.copy()
        actor_image = actor_sprites[actor_frame_index].copy()
        if moving:
            actor_frame_index = (actor_frame_index + 1) % len(actor_sprites)
        
        if actor_direction == 'left':
            actor_image = actor_image.transpose(Image.FLIP_LEFT_RIGHT)
          
        updated_scene.paste(actor_image, tuple(actor_position), actor_image)

        for i, npc_pos in enumerate(npc_positions):
            npc_image = Image.fromarray(npc_anim[i][0]).convert("RGBA")
            npc_image = load_and_prepare_image(npc_image, scale_factor=0.3)
            updated_scene.paste(npc_image, tuple(npc_pos), npc_image)
        
        updated_scene_np = np.array(updated_scene)
        updated_scene_cv2 = cv2.cvtColor(updated_scene_np, cv2.COLOR_RGB2BGR)
        updated_scene_pygame = pygame.image.frombuffer(updated_scene_cv2.tobytes(), updated_scene_cv2.shape[1::-1], "RGB")
        
        screen.blit(updated_scene_pygame, (0, 0))
        pygame.display.flip()

        clock.tick(30)  # Limit to 30 FPS

    pygame.quit()


# Run the update function
update()