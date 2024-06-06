import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


img_size = 128

def load_image(path, size=(img_size, img_size)):
    return Image.open(path).convert('RGB').resize(size)


def save_image(image_array, path):
    image = Image.fromarray(np.uint8(image_array))
    image.save(path)


def interpolate_images(image1, image2, alpha):
    """Interpolates between two images with a given alpha."""
    if image1.size != image2.size:
        raise ValueError("Images must have the same dimensions")
    
    array1 = np.array(image1)
    array2 = np.array(image2)
    
    interpolated_array = (1 - alpha) * array1 + alpha * array2
    return np.clip(interpolated_array, 0, 255)


def generate_interpolated_frames(image1, image2, num_frames, output_dir):
    """Generate a sequence of interpolated images between two images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(num_frames + 1):
        alpha = i / num_frames
        interpolated_image_array = interpolate_images(image1, image2, alpha)
        frame_path = os.path.join(output_dir, f'frame_{i:03d}.jpg')
        save_image(interpolated_image_array, frame_path)


# Load images
image1 = load_image('../../artwork/animation_collision_training_imgs/sprite-sheet-frame-00.png')
image2 = load_image('../../artwork/animation_collision_training_imgs/sprite-sheet-frame-01.png')

# Generate interpolated frames
num_frames = 2  # Number of frames between the two images
output_dir = 'interpolated_frames'
generate_interpolated_frames(image1, image2, num_frames, output_dir)

# Display the frames as a sequence
fig, axes = plt.subplots(1, num_frames + 1, figsize=(15, 5))
for i, ax in enumerate(axes):
    frame_path = os.path.join(output_dir, f'frame_{i:03d}.jpg')
    frame_image = Image.open(frame_path)
    ax.imshow(frame_image)
    ax.axis('off')
plt.show()
