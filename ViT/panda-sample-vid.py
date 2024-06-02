#!/usr/local/bin/python3

import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from VisionTransformer import VisionTransformer

# setup
device = torch.device('cpu')

# sample sequence length
num_images = 10

from Hyperparameters import Hyperparameters
hp = Hyperparameters()

img_size = hp.img_size
patch_size = hp.patch_size
in_channels = hp.in_channels
emb_dim = hp.emb_dim
num_heads = hp.num_heads
num_layers = hp.num_layers
forward_expansion = hp.forward_expansion
num_classes = hp.num_classes
batch_size = hp.batch_size

def load_model(checkpoint_path):
    model = VisionTransformer(
        in_channels=in_channels,
        patch_size=patch_size,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        forward_expansion=forward_expansion,
        num_classes=num_classes,
        img_size=img_size   
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def read_images(image_dir, indices, prefix):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    images = []
    for idx in indices:
        img_path = os.path.join(image_dir, f'{prefix}{idx:05d}.png')
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)
    return torch.stack(images)


def read_images_as_PIL(image_dir, indices, prefix):
    images = []
    for idx in indices:
        img_path = os.path.join(image_dir, f'{prefix}{idx:05d}.png')
        img = Image.open(img_path).convert('RGB')
        images.append(img)
    return images


def display_and_save_plot(source_images, target_images, output_images, epoch):
    fix, axes = plt.subplots(10, 3, figsize=(15, 30))
    for i in range(10):
        axes[i, 0].imshow(transforms.ToPILImage()(source_images[i]))
        axes[i,0].set_title('Source')
        axes[i, 1].imshow(transforms.ToPILImage()(target_images[i]))
        axes[i,1].set_title('Target')
        axes[i, 2].imshow(transforms.ToPILImage()(output_images[i].view(in_channels, img_size, img_size)))
        axes[i,2].set_title('Output')
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'panda3d/mepoch-{epoch:05d}.png')
    plt.close()


import cv2
import glob


def tensor_to_numpy(tensor):

    np_tensor = tensor.detach().cpu().numpy()

    if np_tensor.shape[0] == 1:
        print(f"Squeezing...")
        np_tensor.np.squeeze(1, axis=0)

    if np_tensor.ndim == 3 and np_tensor.shape[0] == 3:
        print("Transposing...")
        np_tensor = np.transpose(np_tensor, (1,2,0))

    np_tensor = (np_tensor * 255).astype(np.uint8)
    return np_tensor




def create_video_from_images(images, output_video_path, fps=30):
    #if not image_files:
    #    print("No images provided.")
    #    return

    # Read the first image to get the dimensions
    frame = images[0]
    if frame is None:
        print(f"Unable to read image file: {image_files[0]}")
        return
    else:
        print(f"Image frame is fine")

    height = img_size
    width = img_size

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in images:
        # frame = cv2.imread(image_file)
        if frame is None:
            print(f"Unable to read frame: {frame}")
            continue
        out.write(np.array(frame))  # Write the frame to the video

    out.release()  # Release the video writer
    print(f"Video saved as {output_video_path}")


def combine_videos_horizontally(video1_path, video2_path, video3_path, output_path):

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    cap3 = cv2.VideoCapture(video3_path)

    if not (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
        print(f'One of the input videos is bad')

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    
    print(f'Width: {width} Height:{height} fps: {fps}')

    border_thickness = 2
    border_color = (0, 0, 255)

    output_width = width * 3 + border_thickness * 4
    output_height = height + border_thickness * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    print(f'Setup complete...')

    idx = 0
    while True:
        print(f'Frame: {idx}')
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        if not (ret1 and ret2 and ret3):
            break

        frame1_bordered = cv2.copyMakeBorder(frame1, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)
        frame2_bordered = cv2.copyMakeBorder(frame2, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)
        frame3_bordered = cv2.copyMakeBorder(frame3, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)

        combined_frame = np.concatenate((frame1_bordered, frame2_bordered, frame3_bordered), axis=1)
        
        out.write(combined_frame)
        idx +=1

    cap1.release()
    cap2.release()
    cap3.release()
    out.release()
    print(f"Video saved as {output_path}")


def main(P):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # checkpoints
    checkpoint_dir = 'panda3d'
    dataset_dir = '../panda3d/frame_caps/naked'
    output_dir = 'output/panda_sim'
    
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if 'main' in f])
    print(f"Using epoch {checkpoints[-1]}")
    checkpoint_path = f'{checkpoint_dir}/{checkpoints[-1]}'

    model = load_model(checkpoint_path)
    model.eval()

    # upper bound
    M = len(os.listdir(dataset_dir)) - P
    # lower bound
    N = 10 
    # stutter length
    Q = 5
    rounded_random_int = round(random.randint(N, M) / Q) * Q
    source_indices = [rounded_random_int + Q * i for i in range(P)]
    target_indices = [value + Q for value in source_indices]

    prefix = 'frame-'
    source_images = read_images(dataset_dir, source_indices, prefix)
    target_images = read_images(dataset_dir, target_indices, prefix)

    source_images = source_images.unsqueeze(0)

    print(f'source images dims: {source_images.shape}')

    for i in range(P):

        with torch.no_grad():
            output_img = model(source_images[:,i,:,:,:]).cpu().view(in_channels, img_size, img_size)

        # Save the output image
        output_img_pil = transforms.ToPILImage()(output_img)
        output_img_path = os.path.join(output_dir, f'oframe_{i:05d}.png')
        output_img_pil.save(output_img_path)

        # Use the output as the new input
        # input_img = output_img.unsqueeze(0)
        if i % 10 == 0:
            print(f"Index: {i:05d}")

    print("Generated 1000 frames.")

    # Generate an MP4 from the resulting output images
    frame_rate = 30  # Define the frame rate
    output_video_path = 'panda3d/output-vid.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (img_size, img_size))

    for i in range(P):
        frame_path = os.path.join(output_dir, f'oframe_{i:05d}.png')
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"MP4 video saved to {output_video_path}.")

    # dump out the other two
    source_images = read_images_as_PIL(dataset_dir, source_indices, prefix)
    target_images = read_images_as_PIL(dataset_dir, target_indices, prefix)
    create_video_from_images(source_images, 'panda3d/source-vid.mp4')
    create_video_from_images(target_images, 'panda3d/target-vid.mp4')

    # now combine them into one mp4
    # this doesn't work, file generates but won't play
    combine_videos_horizontally('panda3d/source-vid.mp4', 'panda3d/target-vid.mp4', 'panda3d/output-vid.mp4', 'panda3d/combo-vid.mp4')    
    # do this in terminal
    # ffmpeg -i panda3d/source-vid.mp4 -i panda3d/target-vid.mp4 -i panda3d/output-vid.mp4 -filter_complex hstack=inputs=3 output.mp4

if __name__ == "__main__":
    P = 1000
    main(P)

