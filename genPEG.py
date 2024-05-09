#!/usr/bin/python3

from PIL import Image
import numpy as np
from scipy import fft

# read jpg image
img_file = "artwork/arena_training_images/racer-track-0040.png"
img = Image.open(img_file).convert('L')

# scale to 256x256
img = img.resize((256,256))
img.show()
img.save("artwork/dct_test/racer-track-0040-grey.jpg")

# convert to greyscale
pass

# convert to Yuv (unneeded for greyscale)
pass

# normalize to -127:127

img_stepdown = np.array(img, dtype=np.float32)
print(img_stepdown)
img_stepdown = img_stepdown - 128.
print(img_stepdown)

# perform DCT on 8x8

def block_dct(block):
    return fft.dct(fft.dct(block.T, norm='ortho').T, norm='ortho')

# quantize to 4x4

def extract_4x4(dct_block):
    return dct_block[:4, :4]

# reconstruct 256px with replication

dct_image = np.zeros(shape=(256, 256))
for i in range(0, 256, 8):
    for j in range(0, 256, 8):
        block = img_stepdown[i:i+8, j:j+8]
        print("Block")
        print(block)
        dct_block = block_dct(block)
        print("DCT Block")
        print(dct_block)
        sub_matrix = extract_4x4(dct_block)
        for x in range(0, 8, 4):
            for y in range(0, 8, 4):
                dct_image[i+x:i+x+4, j+y:j+y+4] = sub_matrix
                
# save with .gjpg extension

new_image = Image.fromarray(dct_image.clip(-128,127)+128).convert('L')
new_filename = "artwork/dct_test/racer-track-0040-genPEG.jpg"
new_image.save(new_filename)
new_image.show()

# do the reverse...pull a frame from the interpolation video and jpeg it

def reverse_process_image(file_path):
    # Step 1: Read the modified image
    img = Image.open(file_path).convert('L')
    data = np.array(img, dtype=np.float32) - 128  # adjust to original normalization

    # Prepare to reconstruct the original image data
    original_data = np.zeros((256, 256))

    # Step 2 & 3: Unreplicate DCT blocks and perform IDCT
    def block_idct(block):
        return fft.idct(fft.idct(block.T, norm='ortho').T, norm='ortho')

    for i in range(0, 256, 8):
        for j in range(0, 256, 8):
            # Step 2: Since the image is constructed from replicated 4x4 DCT blocks, extract the first 4x4 block
            block_4x4 = data[i:i+4, j:j+4]
            # Pad it back to 8x8
            padded_block = np.zeros((8, 8))
            padded_block[:4, :4] = block_4x4

            # Step 3: Perform IDCT on the padded block
            original_block = block_idct(padded_block)
            original_data[i:i+8, j:j+8] = original_block

    # Step 4: Normalize the pixel values back to 0-255
    original_image = Image.fromarray((original_data.clip(-127, 127) + 128).astype(np.uint8))

    # Step 5: Save the reconstructed image
    reconstructed_filename = file_path.replace('.jpg', '_reconstructed.jpg')
    original_image.save(reconstructed_filename)

    return reconstructed_filename

import cv2  # Importing the OpenCV library

def extract_frame(video_path, frame_number):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Set the frame position
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    success, frame = video.read()  # Read the frame
    if success:
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Save the frame as an image file
        cv2.imwrite(f"extracted_frame_{frame_number}.jpg", frame)
        print(f"Frame {frame_number} extracted successfully.")
    else:
        print(f"Error: Could not extract frame {frame_number}.")
    
    video.release()  # Release the video source

# Usage example
video_path = 'artwork/dct_test/racer-track-genPEG.mp4'
frame_number = 150  # Specify the frame number you want to extract
extract_frame(video_path, frame_number)

frame_num = 3
extract_frame(video_path, frame_num)
extract_filename = f'extracted_frame_{frame_num}.jpg' 
reverse_process_image(extract_filename)