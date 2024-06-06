#!/usr/bin/python3
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random

num_frames = 800
left_right_delta = 2


def create_text_overlay(text):
    """Create an image with a bright yellow text overlay."""
    # Create an image with transparent background
    img = Image.new('RGBA', (50, 50), (255, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Use a truetype font
    # font = ImageFont.truetype("arial.ttf", 40)
    font = ImageFont.load_default()
    # Position the text in the top corner
    text_position = (10, 5) if text == 'L' else (40, 5)
    draw.text(text_position, text, font=font, fill=(255, 255, 0, 0))
    return img


def overlay_text(frame, text_image, position):
    """Overlay text image on the frame at the given position."""
    frame.paste(text_image, position, text_image)
    return frame


def collision_detect(frame, overlay_img, position):
    """Check if the overlay is more than halfway on a green background."""
    print(f"Running collision detection")
    green_threshold = np.array([0, 18, 0], dtype=np.uint8)  # Define lower bound of green
    frame_area = np.array(frame)
    print(f"Position: {position}")
    collision_result = 0
    if position[0] < 64:
        print("Collision Left!")
        collision_result = 1
    if position[0] > 196:
        print("Collision Right!")
        collision_result = 2
    return collision_result


def collision_detect1(frame, overlay_img, position):
    """Check if the overlay is more than halfway on a green background."""
    print(f"Running collision detection")
    green_threshold = np.array([0, 18, 0], dtype=np.uint8)  # Define lower bound of green
    frame_area = np.array(frame)
    print(f"Position: {position}")
    overlay_area = frame_area[position[1]:position[1]+overlay_img.height, position[0]:position[0]+overlay_img.width]
    print(f"Overlay Area: {overlay_area}")
    green_pixels = np.all(overlay_area > green_threshold, axis=2)
    green_count = np.sum(green_pixels)

    required_count = (overlay_img.width * overlay_img.height) / 2
    print(f"Required: {required_count} Green Count: {green_count}")
    if green_count >= required_count:
        print("Collision!")
        return True
    return False


def collision_detect2(frame_bgr, overlay_img, position):
    """Check if the overlay is more than halfway on a green background."""
    print(f"Running collision detection")

    # Ensure position does not exceed frame boundaries
    x, y = position
    h, w = overlay_img.height, overlay_img.width
    h, w = frame_bgr.height, frame_bgr.width
    if y + h > frame_bgr.shape[0] or x + w > frame_bgr.shape[1]:
        print(f"X: {x} Y: {y} H: {h} W: {w}")
        return False  # Overlay exceeds frame boundary

    # Convert the relevant area of the frame to HSV
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    overlay_area_hsv = frame_hsv[y:y+h, x:x+w]
    
    # Define 'green' in HSV
    lower_green = np.array([40, 40, 40])  # Hue from 40 to 80 is generally considered green
    upper_green = np.array([80, 255, 255])
    
    # Create a mask that identifies only the green areas within the range
    green_mask = cv2.inRange(overlay_area_hsv, lower_green, upper_green)
    
    # Calculate how many pixels fall within the green range
    green_count = np.sum(green_mask > 0)
    
    # Determine the number of pixels that need to be green for the condition to be true
    required_count = (w * h) / 2

    print(f"Green count: {green_count}  Required count: {required_count}")
    
    return green_count >= required_count


def add_noise(image):
    """Add random percent noise to the non-transparent parts of the image."""
    np_image = np.array(image)
    alpha_channel = np_image[:, :, 3] if np_image.shape[2] == 4 else None

    # Generate noise as a percentage of current pixel values
    noise_scale_factor = np.random.uniform(0.90, 1.10, (np_image.shape[0], np_image.shape[1], 1))
    
    # Apply noise where alpha channel is not zero if it exists
    if alpha_channel is not None:
        mask = np.repeat((alpha_channel[:, :, np.newaxis] > 0), 3, axis=2)
        np_image[:, :, :3] = np_image[:, :, :3] * noise_scale_factor * mask
    else:
        np_image[:, :, :3] = np_image[:, :, :3] * noise_scale_factor

    # Ensure we clip values to maintain valid image data
    np_image = np.clip(np_image, 0, 255).astype(np.uint8)

    return Image.fromarray(np_image, 'RGBA' if alpha_channel is not None else 'RGB')


def process_video_frames(video_path, overlay_image_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Load the overlay image
    original_overlay_img = Image.open(overlay_image_path).convert("RGBA")

    # Scale down the original overlay image by 4x using LANCZOS filter
    new_size = (original_overlay_img.width // 4, original_overlay_img.height // 4)
    overlay_img = original_overlay_img.resize(new_size, Image.LANCZOS)

    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_index = 0
    shift_count = 0
    horizontal_shift = 0
    shift_direction = 0  # 0 for right, 1 for left
    spinning_frames = 0
    spin_angle = 0

    while frame_index < num_frames:  # Limit to first X frames
        ret, frame = cap.read()
        if not ret:
            break  # No more frames or can't fetch frames

        # Convert the frame to RGB and resize
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.resize((256, 256), Image.LANCZOS)

        # Calculate position to place the overlay image (centered near the bottom)
        base_position = (pil_image.width - overlay_img.width) // 2
        position = (base_position + horizontal_shift, pil_image.height - overlay_img.height - 10)

        if shift_count == 0:
    
            shift_count = random.randint(2, 5)  # Determine the next shift interval
            shift_direction = random.randint(0, 5)  # Determine the shift direction

            collision_status = collision_detect(frame, overlay_img, position)
            if collision_status == 1:
                # collision on left...force right
                shift_direction = 2
            if collision_status == 2:
                # collision on right...force left
                shift_direction == 1

            if shift_direction == 1:
                # Flip overlay for left shift
                overlay_img = overlay_img.transpose(Image.FLIP_LEFT_RIGHT)
                overlay_text = 'L'
            elif shift_direction == 2:
                # Flip overlay for right shift
                overlay_img = overlay_img.transpose(Image.FLIP_LEFT_RIGHT)
                overlay_text = 'R'
            else:
                # Normal orientation for right shift
                overlay_img = overlay_img
                overlay_text = ''
        
        # print(f"Direction: {overlay_text}  Shift Count: {shift_count}")

        # Create a blank image with an alpha layer
        blank_image = Image.new("RGBA", pil_image.size)
        
        # Composite the images

        # Character
        composite_image = Image.alpha_composite(pil_image.convert("RGBA"), blank_image)
        composite_image.paste(overlay_img, position, overlay_img)

        # text
        text_position = (0, 0) if overlay_text == 'L' else ((pil_image.width - 50), 0)
        text_overlay = create_text_overlay(overlay_text)
        composite_image.paste(text_overlay, text_position, text_overlay)
        #final_image = overlay_text(composite_image, text_overlay, text_position)

        # Convert back to BGR for OpenCV compatibility if needed
        final_image = cv2.cvtColor(np.array(composite_image), cv2.COLOR_RGBA2BGRA)

        # Save the image
        cv2.imwrite(f"{output_folder}/frame_{frame_index:03d}.png", final_image)
        frame_index += 1
        shift_count -= 1

        # Update horizontal shift
        if shift_direction == 2:  # Shift right
            horizontal_shift += left_right_delta
        if shift_direction == 1:
            horizontal_shift -= left_right_delta
        if shift_direction == 0:
            pass

        # Limit shift to stay within frame boundaries
        horizontal_shift = max(min(horizontal_shift, (pil_image.width - overlay_img.width) // 2), -(pil_image.width - overlay_img.width) // 2)

    cap.release()
    print("Processing complete.")


def generate_video(output_folder, frame_count):
    img_array = []
    for i in range(frame_count):
        img = cv2.imread(f"{output_folder}/frame_{i:03d}.png")
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(f"{output_folder}/output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Video generated successfully.")


# Usage example:
process_video_frames("artwork/racer-track-arena-demo.mp4", "artwork/racer-scooter-red-256px.png", "output")
generate_video("output", num_frames)

