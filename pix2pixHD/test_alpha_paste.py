import numpy as np
from PIL import Image

foreground = Image.open("datasets/car-goes-left/train_frames/frame-000001.jpg")
foreground_rgba = foreground.convert("RGBA")

background = Image.open("datasets/road-rage/train_frames/frame-000001.jpg")
background_rgba = background.convert("RGBA")

mask = foreground.split()[0]
mask = mask.point(lambda p: 255 if p > 5 else 0)
inverted_mask = Image.eval(mask, lambda p: 255 - p)

foreground_rgba.putalpha(mask)

composite_image = Image.new("RGBA", background.size)
composite_image.paste(background, (0,0))
composite_image.paste(foreground_rgba, (0,0), foreground_rgba)

composite_image.show()

