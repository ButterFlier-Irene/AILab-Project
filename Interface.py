import tkinter as tk
from tkinter import *
from tkinter import ttk
import cv2 
from PIL import Image, ImageTk 

window = Tk()  # Sets up GUI

window.title("INTERFACE")  # Titles GUI
window.geometry("1000x1000")  # Sizes GUI
window.configure(bg="black")

btn=Button(window, text="BACK",fg='black')
btn.place(x=950, y=200)
btn1=Button(window, text="EXIT",fg='black', command=window.destroy)
btn1.place(x=950, y=250)
lbl=Label(window, text="KIDS MODE", fg='white', font=("Helvetica", 16), background='black')
lbl.place(x=450, y=3)

vid = cv2.VideoCapture(0) 
width, height = 800, 400
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
window.bind('<Escape>', lambda e: window.quit()) 
label_widget = Label(window) 
label_widget.pack() 
def open_camera(): 
	_, frame = vid.read() 
	opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
	captured_image = Image.fromarray(opencv_image) 
	photo_image = ImageTk.PhotoImage(image=captured_image)  
	label_widget.photo_image = photo_image 
	label_widget.configure(image=photo_image) 
	label_widget.after(10, open_camera) 

button1 = Button(window, text="Open Camera", command=open_camera) 
button1.pack(side='left',expand=True) 


tk.mainloop()

