import os
from PIL import Image

def post_process(root_dir):
    naked_dir = f'{root_dir}/naked'
    overlay_dir = f'{root_dir}/overlay'
    combined_dir = f'{root_dir}/combined'
    # get list of naked dir
    naked_images = sorted([f for f in os.listdir(naked_dir) if f.lower().endswith(('.png'))])
    # get list of overlay dir
    overlay_images = sorted([f for f in os.listdir(overlay_dir) if f.lower().endswith(('.png'))])
    # combine them sequentially
    idx = 0
    for (n_img, o_img) in zip(naked_images, overlay_images):
        print(f'Index: {idx}')
        naked_image = Image.open(f'{naked_dir}/{n_img}').convert('RGBA')
        overlay_image = Image.open(f'{overlay_dir}/{o_img}').convert('RGBA')
        combined_image = Image.alpha_composite(naked_image, overlay_image)
        combined_image.save(f'{combined_dir}/combined-frame-{idx:05f}.png')
        idx += 1

post_process('frame_caps')

