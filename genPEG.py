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

