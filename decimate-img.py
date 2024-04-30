#!/usr/bin/python3

# accept a long image and slice it into frames, in one pixle increments

from PIL import Image

# open the image
im = Image.open("track-128px.png")
print("Opened image")

# get the image dimensions
x,y = im.size
print("Image size : WxH: ", x, " ", y)

# assume 128x128
# remember images are 0,0 in upper left and we're starting from the bottom

# start at bottom, go until height is same as width
for i in range(y-1, x-1, -1): 
    print("Step: ", i-x)
    # left, upper, right, lower
    bbox = (0, i-x, 128, i)
    print("BBox: ", bbox)
    crop_img = im.crop(bbox)
    filename = "racer-track-{:04d}.png".format(i-128)
    print(filename)
    crop_img.save(filename)