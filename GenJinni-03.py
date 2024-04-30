import os
import shutil
from tkinter import Tk, filedialog, Label, Button, Frame, DISABLED, NORMAL, BOTH, Text, Scrollbar, END, BooleanVar, Radiobutton
from tkinter.ttk import Separator
from PIL import Image, ImageTk
import sys
import subprocess
import threading
from tkinter.font import Font

class ConsoleOutput:
    """A class to redirect console output to a tkinter Text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        """Writes a message to the Text widget and scrolls to the bottom."""
        self.text_widget.insert(END, message)
        self.text_widget.see(END)

    def flush(self):
        """Flush does nothing, but is necessary for sys.stdout replacement."""
        pass

def upload_image(category):
    # Set up the directory path for the specific category
    category_dir = os.path.join("uploaded_images", category.lower())

    # Ensure the category-specific directory exists
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

    # Ask the user to select images
    file_paths = filedialog.askopenfilenames(
        title=f"Select {category} images", 
        filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg;*.jpeg")]
    )

    # If Decimate is selected and it's the Arena category, run decimate_image
    if category == "Arena" and decimate_video.get():
        if len(file_paths) > 1:
            print("Only one file can be uploaded for Arena with Decimate selected.")
            file_paths = file_paths[:1]

        # Stub out decimate_image
        decimate_image(file_paths[0])

    # Clear any previous images in the row frame
    for widget in category_frames[category]["images"].winfo_children():
        widget.destroy()

    # Move the selected images to the subfolder and display them
    for path in file_paths:
        # Copy the image to the subfolder
        dest_path = os.path.join(category_dir, os.path.basename(path))
        shutil.copy(path, dest_path)

        # Display the image in the GUI window
        img = Image.open(dest_path)
        img = img.resize((150, 150))  # Resize for display
        img = ImageTk.PhotoImage(img)

        lbl = Label(category_frames[category]["images"], image=img)
        lbl.image = img  # Keep a reference to avoid garbage collection
        lbl.pack(side="left")

    # Change the upload button text
    category_frames[category]["button"].config(text=f"Uploaded: {category}")

    # Check if all categories have images, if so, enable the review button
    if all(os.listdir(os.path.join("uploaded_images", cat.lower())) for cat in category_frames):
        review_button.config(state=NORMAL)


def decimate_image(file_path):
    """Stub function for decimating an image."""
    from PIL import Image

    # Open the image to get its dimensions
    with Image.open(file_path) as img:
        width, height = img.size

    # Write the information to stdout
    sys.stdout.write(f"Decimating the image at {file_path} (dimensions: {width}x{height})...\n")

    # perform the actual decimation
    # open the image
    im = Image.open(file_path)
    sys.stdout.write("Opened image")

    # get the image dimensions
    x, y = im.size
    sys.stdout.write(f"Image size : {x} x {y}")
    sys.stdout.write("Training images go to artwork/arena_training_images/")

    # assume square
    # remember images are 0,0 in upper left and we're starting from the bottom

    # start at bottom, go until height is same as width
    for i in range(y-1, x-1, -1): 
        print("Step: ", i-x)
        # left, upper, right, lower
        bbox = (0, i-x, 128, i)
        print("BBox: ", bbox)
        crop_img = im.crop(bbox)
        frame_name = "racer-track-{:04d}.png".format(i-128)
        filename = "artwork/arena_training_images/racer-track-{:04d}.png".format(i-128)
        print(filename)
        crop_img.save(filename)
        sys.stdout.write(f"Saving frame {frame_name}")


def create_gui():
    global root, review_button, category_frames, decimate_video
    # Set up the main window
    root = Tk()
    root.title("Image Uploader")

    # Dictionary to store frames corresponding to each category
    category_frames = {
        "Arena": {}, 
        "Character": {}, 
        "Collisions": {},
    }

    # Create necessary directories
    for category in category_frames:
        category_dir = os.path.join("uploaded_images", category.lower())
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

    # Create frames and buttons for each category
    for idx, category in enumerate(category_frames):
        category_frame = Frame(root, borderwidth=2, relief="solid")
        category_frame.grid(row=idx*2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Upload button
        category_frames[category]["button"] = Button(category_frame, text=f"Upload {category} Images", command=lambda c=category: upload_image(c))
        category_frames[category]["button"].pack(padx=5, pady=5)

        # Frame for displaying images
        category_frames[category]["images"] = Frame(category_frame)
        category_frames[category]["images"].pack(padx=5, pady=5)

        # If this is the Arena category, add a Decimate option
        if category == "Arena":
            decimate_video = BooleanVar()

            # Radio button for Decimate
            decimate_button = Radiobutton(category_frame, text="Decimate", variable=decimate_video, value=True)
            decimate_button.pack(padx=5, pady=5)

        # Add a separator line between each category
        if idx < len(category_frames) - 1:
            Separator(root, orient="horizontal").grid(row=idx*2+1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    # Create a review button
    review_button = Button(root, text="Review the Setup", state=DISABLED, command=review_setup)
    review_button.grid(row=len(category_frames)*2, column=0, columnspan=2, padx=5, pady=10)

    # Create a text box for console output
    console_frame = Frame(root)
    console_frame.grid(row=len(category_frames)*2+1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    # Define a smaller font
    console_font = Font(family="Courier", size=10)  # Adjust font family and size as needed

    console_text = Text(console_frame, height=15, wrap="word", font=console_font)
    console_text.pack(side="left", fill=BOTH, expand=True)

    scrollbar = Scrollbar(console_frame, command=console_text.yview)
    scrollbar.pack(side="right", fill="y")
    console_text.config(yscrollcommand=scrollbar.set)

    # Redirect stdout to the console text widget
    sys.stdout = ConsoleOutput(console_text)

    # Start the GUI event loop
    root.mainloop()

def review_setup():
    """Runs the external Python script 'build-the-game.py' and pipes its output into the console text widget."""
    print("Running the external script 'build-the-game.py'...")

    # Run the script in a new thread
    def run_script():
        # Run the script and capture its output
        process = subprocess.Popen(
            ["python3", "build-the-game.py"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )

        # Stream the output into the console text widget
        for line in process.stdout:
            sys.stdout.write(line)

        # Check for errors
        for line in process.stderr:
            sys.stdout.write(f"ERROR: {line}")

    threading.Thread(target=run_script).start()

# Run the GUI
if __name__ == "__main__":
    create_gui()
