from tkinter import *
from tkinter import filedialog
from tkinter import GROOVE, Button, Frame, Label, Tk
from handLandmarks import GetLandmarks, DrawLandmarks, DrawBoundingBox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import random
import pandas as pd
import os

def detect_image_gui(tk_win: Tk):
    
    # Set the title of the main window
    tk_win.title('ASL Alphabet Recognition')
    
    # Get the screen width and height
    width = tk_win.winfo_screenwidth()
    height = tk_win.winfo_screenheight()
    print("TKinter window size:", width, height)
    
  
    # Set the geometry of the main window to fill the entire screen
    tk_win.geometry("%dx%d" % (width, height))

    Label(tk_win, text="Label 1").grid(row=0, column=1)
    Label(tk_win, text="Label 2").grid(row=1, column=1)
    Label(tk_win, text="Label 1").grid(row=2, column=1)
    
    # Create a frame that fills the entire window with a specific background color
    frame_1 = Frame(tk_win, width=width, height=height,bg="#494848").place(x=0, y=0)
    label_widget_video = Label(frame_1)
    label_widget_video.place(x=30, y=140)

    mylabel1 = Label(
    tk_win,
    text='ASL Alphabet Recognition',
    font=('Helvetica', 26, 'bold'),
    bd=5,
    bg='#b4b4b4',
    fg='#2c2c2c',
    relief=GROOVE,
    width=43).place(x=200,y=20)

    # Create a Listbox to display the history
    label2=Label(tk_win,
    text='history:',
    font=('Helvetica', 16, 'bold'),
    bd=5,
    bg='#b4b4b4',
    fg='#2c2c2c',
    relief=GROOVE,
    width=20).place(x=835,y=100)
    listbox = Listbox(tk_win, width=90, height=20)
    listbox.place(x=690, y=140)

    label3=Label(tk_win, # Create a Listbox to display the webcam
    text='webcam:',
    font=('Helvetica', 16, 'bold'),
    bd=5,
    bg='#b4b4b4',
    fg='#2c2c2c',
    relief=GROOVE,
    width=18).place(x=180,y=100)

    # Add a line in canvas widget
    our_canvas=Canvas(tk_win,width=1,height=1000,bg="white").place(x=640,y=73)
    our_canvas1=Canvas(tk_win,width=640,height=1,bg="white").place(x=640,y=500)
    our_canvas2=Canvas(tk_win, width=8,height=8,bg='black').place(x=242,y=114)

    def run():
        our_canvas2=Canvas(tk_win, width=8,height=8,bg='red').place(x=242,y=114)
        btn2=Button(tk_win, text="WEBCAM OFF",command=lambda:camoff(),fg='white', bg='#75706f',width=20, height=5).place(x=690,y=540)
        detect_signs(tk_win, label_widget_video)
    
    def camoff():
        label_widget_video.destroy()
        our_canvas2=Canvas(tk_win, width=8,height=8,bg='black').place(x=242,y=114)
        btn2=Button(tk_win, text="WEBCAM ON",fg='white',bg='#75706f', command=run,width=20, height=5).place(x=690,y=540)
      
    btn1=Button(tk_win, text="EXIT",fg='white',bg='#75706f', command=tk_win.destroy,width=20, height=5).place(x=1100,y=540)
    btn=Button(tk_win, text="KIDS MODE",command=lambda:go_on(), fg='white',bg='#75706f',width=20, height=5).place(x=900,y=540)
    btn2=Button(tk_win, text="WEBCAM ON",fg='white',bg='#75706f', command=run,width=20, height=5).place(x=690,y=540)

    def go_back():
        btn=Button(tk_win, text="KIDS MODE",command=lambda:go_on(), fg='white',bg='#75706f',width=20, height=5).place(x=900,y=540) 
        mylabel1 = Label(
        tk_win,
        text='ASL Alphabet Recognition',
        font=('Helvetica', 26, 'bold'),
        bd=5,
        bg='#b4b4b4',
        fg='#2c2c2c',
        relief=GROOVE,
        width=43).place(x=200,y=20)

        label2=Label(tk_win,
        text='history:',
        font=('Helvetica', 16, 'bold'),
        bd=5,
        bg='#b4b4b4,',
        fg='#2c2c2c',
        relief=GROOVE,
        width=20).place(x=835,y=100)

    def go_on():
        btn=Button(tk_win, text="BACK",command=lambda:go_back(), fg='white',bg='#75706f',width=20, height=5).place(x=900,y=540)
        mylabel1 = Label(
        tk_win,
        text='KIDS MODE',
        font=('Helvetica', 26, 'bold'),
        bd=5,
        bg='white',
        fg='#374254',
        relief=GROOVE,
        width=43).place(x=200,y=20)

        label5=Label(tk_win,
        text='images:',
        font=('Helvetica', 16, 'bold'),
        bd=5,
        bg='#b4b4b4',
        fg='#2c2c2c',
        relief=GROOVE,
        width=20).place(x=835,y=100)

   # Pack the Top Title on top-center of the window:

def detect_signs(win_tk: Tk,  label_widget_video: Label):
    
    labels = {'0':'0','1': '1', '2': '2', '3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','a':'A','b':'B','c':'C','d':'D','e':'E','f':'F','g':'G','h':'H','i':'I','j':'J','k':'K','l':'L','m':'M','n':'N','o':'O','p':'P','q':'Q','r':'R','s':'S','t':'T','u':'U','v':'V','w':'W','x':'X','y':'Y','z':'Z'}
    
    # Load the pre-trained model
    our_model = joblib.load("model.joblib")
    color = (0,0,0)
    
    # Init videocapture
    cap = cv2.VideoCapture(0)

    print("Camera is on... Entering the loop...")
    
    lab = {'b':'Dataset_ASL/0/hand1_0_bot_seg_2_cropped.jpeg','1': 'Dataset_ASL/1/hand1_1_bot_seg_2_cropped.jpeg', 'v': 'Dataset_ASL/1/hand1_1_bot_seg_2_cropped.jpeg'}

    def getNextLetter(): 
        time.sleep(1)
        return random.choice(list(lab.keys()))
    
    letter = getNextLetter()
    
    data = pd.read_csv('dataset.csv')
    values = dict.fromkeys(set(data.label), 0)
    
    #for item in values:
        
    for k, v in values.items():
        print(k, v)
    
    #Label(tk_win, text="Lalala", anchor="e").grid(row=0, column=0)
    
    #Label(tk_win, text='ASL Alphabet Recognition', font=('Comic Sans MS', 24, 'bold'), bd=5, bg='#20262E', fg='#F5EAEA', relief=GROOVE).grid(row=0, column=0)
    
    while cap.isOpened():
        _, img = cap.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # converting the channels
        coordinates, result = GetLandmarks(img)
        
        #cv2.putText(img,  f"Show Letter {letter.upper()}",  (50, 70),  cv2.FONT_HERSHEY_SIMPLEX, 2,  (0, 255, 255),  2,  cv2.LINE_4) 
        #print(values)
        if coordinates != '':
            img = cv2.cvtColor(DrawLandmarks(img, result), cv2.COLOR_RGB2BGR)
            
            coordinates = np.array(coordinates).reshape(1, 42)
            prediction = our_model.predict(coordinates)
            predicted_character = labels[(prediction[0])]
            
            img = cv2.cvtColor(DrawBoundingBox(img, result, predicted_character,color), cv2.COLOR_RGB2BGR)
            print(predicted_character)
            if prediction == letter:
                print('correct')
                color = (255,255,255)
                v = values.get(letter) + 1
                up_dict = {letter:v}
                #print("Dictionary before updation:",dict)
                values.update(up_dict)
                
                letter = getNextLetter()

        img = cv2.resize(img, None, fx = 0.9, fy = 1.0)
        #cv2.imshow('image' , img)
        # Convert the image to a PIL image
        image_tk = Image.fromarray(img)
        
        # Convert the PIL image to a Tkinter PhotoImage
        final_tk_image = ImageTk.PhotoImage(image_tk)
        
        # Set the image in the TKinter label widget:
        label_widget_video.configure(image=final_tk_image)
        label_widget_video.image = final_tk_image
        
        # make TKinter window to refresh:
        tk_win.update()

    cap.release()

def kidsmode(values):
    pass
    '''
    labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','Y','X','Z']
    kidimages = []

    for labels in os.listdir('PHOTOS'):
        d[labels]=kidimages                   

    d={'label':labels,'img':kidimages}              
    #d[values]=key   
    '''


if __name__ == "__main__":
    # Create the main window
    tk_win = Tk() 
    detect_image_gui(tk_win)

tk_win.mainloop()

    