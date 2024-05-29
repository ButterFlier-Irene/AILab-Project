from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import random
import os
import sys

'''
# everything is a Widget, so create the root widget
root = Tk()

# create a label Widget
myLabel = Label(root, text="Hello World!").grid(row=0, column=5)

# put the label widget in the window(root widget)
# to put stuff (show) on the screen with tkinter, use the pack():
# myLabel.pack() 


# create the constant/main loop to keep the window open
root.mainloop()
'''


# Create the main window
win = Tk()

# Get the screen width and height
width = win.winfo_screenwidth()
height = win.winfo_screenheight()
print(width,height)

# Set the geometry of the main window to fill the entire screen
win.geometry("%dx%d" % (width, height))

# Create a frame that fills the entire window with a specific background color
frame_1 = Frame(win, width=width, height=height, bg="#181823").place(x=0, y=0)

# Set the title of the main window
win.title('ASL Alphabet Recognition')

# Create a label with specific text, font, background, foreground, border, and other properties
mylabel1 = Label(
    win,
    text='ASL Alphabet Recognition',
    font=('Comic Sans MS', 26, 'bold'),
    bd=5,
    bg='#20262E',
    fg='#F5EAEA',
    relief=GROOVE,
    width=5000
).pack(pady=20, padx=500)


# Create a list to store the history of signs
sign_history = []

# Function to display a new sign and update history
def display_sign():
    # Select a random sign (in a real application, this would be your sign detection logic)
    new_sign = random.choice(labels)
    
    # Add the new sign to the history if it's different from the last one
    if not sign_history or sign_history[-1] != new_sign:
    # Add the new sign to the history
        sign_history.append(new_sign)
    
  # Update the Listbox
        listbox.insert(END, new_sign)

# Create a Listbox to display the history
listbox = Listbox(win, width=50, height=20)
listbox.pack(pady=20)


# Create a button to trigger the display of a new sign
#display_button = Button(win, text="Show Sign", command=display_sign)
display_button = Button(win, text="Show Sign", command = os.system('python detectSignsCamera.py'))
display_button.pack(pady=10)







'''
# Exit feature in GUI:
def lbl():
    global label1
    label1.destroy()
def lbl2():
    global label1
    cv2.destroyAllWindows()
    label1.destroy()
'''    
    
win.mainloop()
