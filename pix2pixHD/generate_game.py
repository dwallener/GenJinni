# modify the generate_video script to combine images and generate gameplay

### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model

from tqdm import tqdm
from PIL import Image
import torch
import shutil
import video_utils
import image_transforms
import random
import numpy as np
from torchvision.transforms import transforms
from scipy.special import expit

# track options
opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# additional enforced options for video
opt.video_mode = True
opt.label_nc = 0
opt.no_instance = True
opt.resize_or_crop = "none"

# loading initial frames from: ./datasets/NAME/test_frames
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# car options
opt_car = TestOptions().parse(save=False)
opt_car.nThreads = 1   # test code only supports nThreads = 1
opt_car.batchSize = 1  # test code only supports batchSize = 1
opt_car.serial_batches = True  # no shuffle
opt_car.no_flip = True  # no flip

# additional enforced options for video
opt_car.video_mode = True
opt_car.label_nc = 0
opt_car.no_instance = True
opt_car.resize_or_crop = "none"

# loading initial frames from: ./datasets/NAME/test_frames
data_loader_car = CreateDataLoader(opt)
dataset_car = data_loader.load_data()

# this directory will contain the generated videos
output_dir = os.path.join(opt.checkpoints_dir, opt.name, 'output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# this directory will contain the frames to build the video
frame_dir = os.path.join(opt.checkpoints_dir, opt.name, 'frames')
if os.path.isdir(frame_dir):
    shutil.rmtree(frame_dir)
os.mkdir(frame_dir)

if opt.png:
    ext = 'png'
else:
    ext = 'jpg'

frame_index = 1
options_text = ""
if opt.start_from == "noise":
    # careful, default value is 1024x512
    t = torch.rand(1, 3, opt.fineSize, opt.loadSize)

elif opt.start_from  == "video":
    # use initial frames from the dataset
    for data in dataset:
        t = data['left_frame']
        video_utils.save_tensor(
            t,
            frame_dir + "/frame-%s.%s" % (str(frame_index).zfill(5), ext),
            text="original video",
        )
        frame_index += 1
else:
    # use specified image
    filepath = opt.start_from
    options_text += ("_" + filepath.split('/')[-1])
    
    if os.path.isfile(filepath):
        t = video_utils.im2tensor(Image.open(filepath))
        
        if(opt.start_frames_before > 0):
            split = filepath.split('-')
            frame_no = split[-1]
            frame_n = int(frame_no.split('.')[0])
        else:
            video_utils.save_tensor(
                t,
                frame_dir + "/frame-%s.%s" % (str(frame_index).zfill(5), ext),
            )
            frame_index += 1
        
        for i in range(opt.start_frames_before):
            
            if(opt.start_frames_before > 0):
                f = frame_n - (opt.start_frames_before-i)
                f = str(f).zfill(6)+'.jpg'
                fpath = filepath.replace((frame_no),f)
                t = video_utils.im2tensor(Image.open(fpath))

            video_utils.save_tensor(
                t,
                frame_dir + "/frame-%s.%s" % (str(frame_index).zfill(5), ext),
            )
            frame_index += 1
    else:
        print('oops, not a file using filepath')

current_frame = t

duration_s = opt.how_many / opt.fps
if (opt.zoom_lvl!=0):
   options_text += ("_with-%d-zoom" % opt.zoom_lvl) 

if(opt.zoom_cres):
    zoom_amt = 0
    options_text += ("_zoom-cres-%d" % opt.zoom_inc)

video_id = "epoch-%s_%s_%.1f-s_%.1f-fps%s" % (
    str(opt.which_epoch),
    opt.name,
    duration_s,
    opt.fps,
    options_text
)

# this just does the track
model = create_model(opt)
# this is for the car
model_car = create_model(opt_car)


# this is the main generation loop
# expand this to the following steps
# 1. generate the next frame of track
# 2. generate the next frame of car based on keyboard input
# 3. composite the car and track
# 4. save that image as the next frame

# other things needed...
# load the car movement models separately

# for centering the car
offset_x = -32
offset_y = 0
prev_key = 's'

for i in tqdm(range(opt.how_many)):

    # this gets us the track frame
    next_frame = video_utils.next_frame_prediction(model, current_frame)
    track_frame = next_frame
    # do this here because we will munge next_frame for video generation
    current_frame = next_frame

    # construct the car frame here
    # read key (or generate random)
    # composite the car frame with key encoding
    # feed that to the frame predictor
    next_key_select = random.randint(0,5)
    if next_key_select == 0:
        # go right
        next_key = 'r'
        if prev_key == 'r':
            offset_x = offset_x + 4
        else:
            offset_x = offset_x + 2
        prev_key = next_key
    elif next_key_select == 1:
        # go left
        next_key = 'l'
        if prev_key == 'l':
            offset_x = offset_x - 4
        else:
            offset_x = offset_x - 2
        prev_key = next_key
    else:
        # go straight
        next_key = 's'
        prev_key = next_key
    
    # do the car frame here
    # next_frame_car = video_utils.netx_frame_prediction(model_car, current_frame)
    # for testing
    next_frame_car = Image.open("datasets/car-goes-left/train_frames/frame-000001.jpg")

    if opt.zoom_lvl != 0:
        next_frame = image_transforms.zoom_in(next_frame, zoom_level=opt.zoom_lvl)

    if(opt.zoom_cres):
        if(i % opt.zoom_inc == 0):
            zoom_amt += 1

        next_frame = image_transforms.zoom_in(next_frame, zoom_level=zoom_amt)


    if opt.heat_seeking_lvl != 0:
        next_frame = image_transforms.heat_seeking(next_frame, translation_level=opt.heat_seeking_lvl, zoom_level=opt.heat_seeking_lvl)

    # combine frames here
    background = expit(next_frame.cpu().detach().numpy()) * 255
    background = background.astype(np.uint8)
    # convert to 0-1 and then to uint8

    background = np.squeeze(background) # remove singleton array
    background = np.moveaxis(background, 0, -1) # move the truple to the back
    
    bg_img = Image.fromarray(background, 'RGB')
    bg_img_rgba = bg_img.convert("RGBA")

    foreground = next_frame_car
    foreground_rgba = foreground.convert("RGBA")

    # mask the encoding out with Alpha channel
    #comp_frame = Image.blend(bg_img, foreground, 0.5)
    # create the alpha mask

    mask = foreground.split()[0]
    mask = mask.point(lambda p: 255 if p > 5 else 0)
    inverted_mask = Image.eval(mask, lambda p: 255 - p)

    foreground_rgba.putalpha(mask)

    composite_image = Image.new("RGBA", bg_img.size)
    composite_image.paste(bg_img, (0,0))
    composite_image.paste(foreground_rgba, (offset_x, offset_y), foreground_rgba)
    #composite_image.show()
    #input("Press a key to continue:")

    #composite_image.show()

    # up to here, this works pretty well
    # the problem now is putting it back as a [1,3,64,64] tensor

    to_tensor = transforms.ToTensor()
    comp_frame = composite_image.convert("RGB")
    next_frame = to_tensor(comp_frame)
    next_frame = next_frame.unsqueeze(0)

    # ok that works
    # still an issue with passing images into video generation that don't have an alpha channel
    # it's muting all the colors

    video_utils.save_tensor(
        next_frame, 
        frame_dir + "/frame-%s.%s" % (str(frame_index).zfill(5), ext),
    )
    #current_frame = next_frame
    print("Generating next frame from model output")
    frame_index+=1

video_path = output_dir + "/" + video_id + ".mp4"
while os.path.isfile(video_path):
    video_path = video_path[:-4] + "-.mp4"

video_utils.video_from_frame_directory(
    frame_dir, 
    video_path, 
    framerate=opt.fps, 
    crop_to_720p=opt.no_crop,
    reverse=False,
)

print("video ready:\n%s" % video_path)
