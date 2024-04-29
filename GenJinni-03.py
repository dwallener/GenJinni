import os
import shutil
from tkinter import Tk, filedialog, Label, Button, Frame, DISABLED, NORMAL, BOTH, Text, Scrollbar, END
from tkinter.ttk import Separator
from PIL import Image, ImageTk
import sys
import subprocess

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

def create_gui():
    global root, review_button, category_frames
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

        # Add a separator line between each category
        if idx < len(category_frames) - 1:
            Separator(root, orient="horizontal").grid(row=idx*2+1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    # Create a review button
    review_button = Button(root, text="Setup is Reviewed and Accpeted", state=DISABLED, command=review_setup)
    review_button.grid(row=len(category_frames)*2, column=0, columnspan=2, padx=5, pady=10)

    # Create a text box for console output
    console_frame = Frame(root)
    console_frame.grid(row=len(category_frames)*2+1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    console_text = Text(console_frame, height=15, wrap="word")
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

# Run the GUI
if __name__ == "__main__":
    create_gui()
