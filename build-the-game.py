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
    for j in range(0,t):
        dummi = dummi + 1


print("Starting processing...\n")
print("First...build the gameplay arena...\n")
print("Interpolating ARENA images into training video...\n")

# time.sleep(3)
threadsafe_pause(300000)

print("Training video: /output/training_video.mp4\n")
print("Decimating training video into training framees...\n")
print("ffmpeg -i output/training_video.mp4 output/training_frames/frame-\%04d.jpg ")
# time.sleep(2)

print("Training frames: /output/training_frames...\n")

print("Training the frame predictor for " + str(arena_epochs) + " epochs...")
for i in range(0, arena_epochs):
    print("Arena frame predictor Epoch ", i, ":")
    print("\n")
    # time.sleep(5)

print("ARENA frame prediction completed")
print("Model at output/models/arena_model.pt")

print("Training collision classifier for ", collision_epochs, " epochs")
for i in range(0, collision_epochs):
    print("Collision classifier Epoch ", i, ":")
    print("\n")
    # time.sleep(5)

print("COLLISION classifier completed\n")
print("Model at output/models/collision_detect_model.pt")

print("Training for COLLISON frame predictor for ", collision_epochs, " epochs")
for i in range(0, collision_epochs):
    print("Collision frame predictor Epoch ", i, ":")
    print("\n")
    # time.sleep(5)

print ("COLLISION encoder completed\n")

print("Encoding character movement as WASD...\n")
# time.sleep(1)
print("...\n")
# time.sleep(1)
print("...\n")
# time.sleep(1)

print("Character movement encoder completed\n")
print("Model at output/models/character_movement.pt")

print("Modifying base script...\n")
# time.sleep(3)
print("Base script modified: output/models/game_script.py\n")

print("Cleaning up...\n")
# time.sleep(2)
print("Generation completed\n")
