#!/usr/bin/python3

from PIL import Image
import numpy as np
from scipy import fft

# read jpg image
img_file = ""
img = Image.open(img_file)

# scale to 256x256
img = img.resize((256,256))

# convert to greyscale
pass

# convert to Yuv (unneeded for greyscale)
pass

# normalize to -127:127
img_stepdown = np.array(img, dtype=np.float32)
img_stepdown = img_stepdown - 128.

# perform DCT on 8x8

def block_dct(block):
    return fft.dct(fft.dct(block.T, norm='ortho').T, norm='ortho')

# quantize to 4x4
def extract_4x4(dct_block):
    return dct_block[:4, :4]

# reconstruct 256px with replication
dct_image = np.array((256, 256))
for i in range(0, 256, 8):
    for j in range(0, 256, 8):
        block = img_stepdown[i:i+8, j:j+8]
        dct_block = img_stepdown(block)
        sub_matrix = extract_4x4(dct_block)
        for x in range(0, 8, 4):
            for y in range(0, 8, 4):
                dct_image[i+x:i+x+4, j+y:j+y+4] = sub_matrix
                
# save with .gjpg extension
new_image = Image.fromarray(dct_image.clip(-128,127)+128).convert('L')
new_filename = ""
new_image.save(new_filename)


