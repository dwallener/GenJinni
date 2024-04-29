#!/usr/local/bin/python3

#import time

# stubbed out...this is where we will actually run everything...

# vars
arena_epochs = 5
collision_epochs = 10
character_epochs = 20
dummi = 0

def threadsafe_pause(t):
    dummi = 0
    for j in range(0, t*10):
        dummi = dummi + 1

print("Starting processing...")
print("First...build the gameplay arena...")
print("Interpolating ARENA images into training video...")

#time.sleep(3)
threadsafe_pause(300000)

print("Training video: /output/training_video.mp4")
print("Decimating training video into training framees...")
print("ffmpeg -i output/training_video.mp4 output/training_frames/frame-\%04d.jpg ")
#time.sleep(2)
threadsafe_pause(300000)
print("Training frames: /output/training_frames...")

print("Training the frame predictor for " + str(arena_epochs) + " epochs...")
for i in range(0, arena_epochs):
    print("Arena frame predictor Epoch ", i, ":")
    #time.sleep(5)
    threadsafe_pause(300000)

print("ARENA frame prediction completed")
print("Model at output/models/arena_model.pt")

print("Training collision classifier for ", collision_epochs, " epochs")
for i in range(0, collision_epochs):
    print("Collision classifier Epoch ", i, ":")
    #time.sleep(5)
    threadsafe_pause(300000)

print("COLLISION classifier completed\n")
print("Model at output/models/collision_detect_model.pt")

print("Training for COLLISON frame predictor for ", collision_epochs, " epochs")
for i in range(0, collision_epochs):
    print("Collision frame predictor Epoch ", i, ":")
    #time.sleep(5)
    threadsafe_pause(300000)

print ("COLLISION encoder completed\n")

print("Encoding character movement as WASD...")
#time.sleep(1)
threadsafe_pause(300000)
print("...")
#time.sleep(1)
threadsafe_pause(300000)
print("...")
#time.sleep(1)
threadsafe_pause(300000)

print("Character movement encoder completed\n")
print("Model at output/models/character_movement.pt")

print("Modifying base script...")
#time.sleep(3)
threadsafe_pause(300000)
print("Base script modified: output/models/game_script.py")

print("Cleaning up...")
#time.sleep(2)
threadsafe_pause(300000)
print("Generation completed")
