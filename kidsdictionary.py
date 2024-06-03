import os
import cv2

# Define the directory path
photos_dir = 'kidsimgs'

# Define the labels
labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# Initialize an empty dictionary to store images
images_dict = {}
def get_kids_dict():
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
            print(item_path, label)
            # Check if the image was successfully read
            if image is not None:
                # Store the image in the dictionary with the corresponding label
                images_dict[label] = image
            else:
                print(f"Failed to read the image file '{item_path}'")
    else:
        print(f"The directory '{photos_dir}' does not exist.")
    

    return images_dict

get_kids_dict()
    
    

