from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import random
import os
import sys
import detectSignsCamera

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
    font=('Helvetica', 26, 'bold'),
    bd=5,
    bg='#20262E',
    fg='#F5EAEA',
    relief=GROOVE,
    width=500).pack(pady=20, padx=200)


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
label2=Label(win,
    text='history:',
    font=('Helvetica', 16, 'bold'),
    bd=5,
    bg='#20262E',
    fg='#F5EAEA',
    relief=GROOVE,
    width=20
).place(x=835,y=100)
listbox = Listbox(win, width=90, height=20)
listbox.place(x=690, y=140)

label3=Label(win, # Create a Listbox to display the webcam
    text='webcam:',
    font=('Helvetica', 16, 'bold'),
    bd=5,
    bg='#20262E',
    fg='#F5EAEA',
    relief=GROOVE,
    width=18
).place(x=180,y=100)
listbox = Listbox(win, bg='gray',width=95, height=30)
listbox.place(x=30, y=140)

# Add a line in canvas widget
our_canvas=Canvas(win,width=1,height=1000,bg="white").place(x=640,y=73)
our_canvas1=Canvas(win,width=640,height=1,bg="white").place(x=640,y=500)
our_canvas2=Canvas(win, width=8,height=8,bg='black').place(x=242,y=114)

def run():
    our_canvas2=Canvas(win, width=8,height=8,bg='red').place(x=242,y=114)

btn1=Button(win, text="EXIT",fg='black', command=win.destroy,width=20, height=5).place(x=1100,y=540)
btn=Button(win, text="KIDS MODE",command=lambda:go_on(), fg='black',width=20, height=5).place(x=900,y=540)
btn2=Button(win, text="WEBCAM ON",fg='black', command=run,width=20, height=5).place(x=690,y=540)

def go_back():
    btn=Button(win, text="KIDS MODE",command=lambda:go_on(), fg='black',width=20, height=5).place(x=900,y=540) 
    mylabel1 = Label(
    win,
    text='ASL Alphabet Recognition',
    font=('Helvetica', 26, 'bold'),
    bd=5,
    bg='#20262E',
    fg='#F5EAEA',
    relief=GROOVE,
    width=43).place(x=200,y=20)

    label2=Label(win,
    text='history:',
    font=('Helvetica', 16, 'bold'),
    bd=5,
    bg='#20262E',
    fg='#F5EAEA',
    relief=GROOVE,
    width=20).place(x=835,y=100)

def go_on():
    btn=Button(win, text="BACK",command=lambda:go_back(), fg='black',width=20, height=5).place(x=900,y=540)
    mylabel1 = Label(
    win,
    text='KIDS MODE',
    font=('Helvetica', 26, 'bold'),
    bd=5,
    bg='white',
    fg='blue',
    relief=GROOVE,
    width=43).place(x=200,y=20)

    label5=Label(win,
    text='images:',
    font=('Helvetica', 16, 'bold'),
    bd=5,
    bg='#20262E',
    fg='#F5EAEA',
    relief=GROOVE,
    width=20).place(x=835,y=100)









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
