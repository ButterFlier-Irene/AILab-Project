import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Define the directory path
photos_dir = 'kidsimgs'

# Define the labels
labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Initialize an empty dictionary to store images
images_dict = {}

# Ensure the directory exists
if os.path.exists(photos_dir):
    # Get the list of files
    files = [f for f in os.listdir(photos_dir) if os.path.isfile(os.path.join(photos_dir, f))]
    
    # Check if there are more files than labels
    if len(files) > len(labels):
        print("Warning: There are more files than labels. Some files will not be labeled.")
    
    # Iterate over the files and labels
    for label, item in zip(labels, files):
        item_path = os.path.join(photos_dir, item)
        
        # Read the image file using OpenCV
        image = cv2.imread(item_path)
        
        # Check if the image was successfully read
        if image is not None:
            # Store the image in the dictionary with the corresponding label
            images_dict[label] = image
        else:
            print(f"Failed to read the image file '{item_path}'")
else:
    print(f"The directory '{photos_dir}' does not exist.")

# Function to display image given a specific label
def show_image(label):
    if label in images_dict:
        image = images_dict[label]
        
        # Convert the image from BGR (OpenCV format) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert the image to a PIL format
        image_pil = Image.fromarray(image_rgb)
        
        return image_pil
    else:
        print(f"Label '{label}' not found in the dictionary.")
        return None

# Tkinter GUI
def get_screen_scaling():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    scaling_factor = min(screen_width / 1920, screen_height / 1080)  # Assuming 1920x1080 as the base resolution
    root.destroy()
    return scaling_factor
root=tk.Tk()
root.title('KIDS MODE')
scaling_factor = get_screen_scaling()
    
# Get the screen width and height
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("TKinter window size:", width, height)
    
  
# Set the geometry of the main window to fill the entire screen
root.geometry("%dx%d" % (width, height))
root.title("Image Display with OpenCV and Tkinter")
frame_1 =tk. Frame(root, width=width, height=height,bg="#494848").place(x=0, y=0)
label_widget_video = tk.Label(frame_1)
label_widget_video.place(x=30, y=140)

def open_img():
    img_pil = show_image('A')  # Change 'A' to any label you want to display
    if img_pil:
        img_pil = img_pil.resize((250, 250), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img_pil)
        panel = tk.Label(root, image=img_tk)
        panel.image = img_tk  # Keep a reference to avoid garbage collection
        panel.place(x=100,y=30)

btn = tk.Button(root, text='Open Image', command=open_img)
btn.pack()

root.mainloop()
