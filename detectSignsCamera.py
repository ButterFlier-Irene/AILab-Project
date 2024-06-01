
from tkinter import GROOVE, Button, Frame, Label, Tk, Canvas,Listbox
from handLandmarks import GetLandmarks, DrawLandmarks, DrawBoundingBox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import random
import pandas as pd
import time
import IPython.display 
#import playsound

def detect_image_gui(tk_win: Tk):
    
    # Set the title of the main window
    tk_win.title('ASL Alphabet Recognition')
    
    # Get the screen width and height
    width = tk_win.winfo_screenwidth()
    height = tk_win.winfo_screenheight()
    print("TKinter window size:", width, height)
    
    # Set the geometry of the main window to fill the entire screen
    tk_win.geometry("%dx%d" % (width, height))

    # Create a frame that fills the entire window with a specific background color
    frame_1 = Frame(tk_win, width=width, height=height,bg="#494848").place(x=0, y=0)
    label_widget_video = Label(frame_1)
    label_widget_video.grid(row = 2, column = 0, sticky = 'w',rowspan=22, columnspan=2)

    mylabel1 = Label(tk_win,text='ASL Alphabet Recognition',font=('Helvetica', 26, 'bold'),bd=5,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE,width=43)
    mylabel1.grid(row = 0, columnspan=5)

    # Create a Listbox to display the history
    label2=Label(tk_win,text='history:',font=('Helvetica', 16, 'bold'),bd=5,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE,width=20)
    label2.grid(row = 1, column = 2,columnspan=3)
    
    # Create a Listbox to display the webcam
    label3=Label(tk_win, text='webcam:',font=('Helvetica', 16, 'bold'),bd=5,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE,width=18)
    label3.grid(row = 1, column = 0,columnspan=2)


    #dataList = [4,5,6,3,7]
    #listbox = Listbox(tk_win, height=20) #right rectangle frame
    #àlistbox.grid(row = 2, column = 2, sticky = 'e', columnspan=2, rowspan=20)  

    data = pd.read_csv('dataset.csv')
    values = dict.fromkeys(set(data.label), 0)

    # Add a line in canvas widget
    #our_canvas=Canvas(tk_win,width=1,height=1000,bg="white").grid(row = 0, column = 0, sticky = 'w')
   # our_canvas1=Canvas(tk_win,width=640,height=1,bg="white").place(x=640,y=500) 
    #our_canvas2=Canvas(tk_win, width=8,height=8,bg='black').place(x=242,y=114)

    def run():
        our_canvas2=Canvas(tk_win, width=8,height=8,bg='red').place(x=242,y=114)
        btn2=Button(tk_win, text="WEBCAM OFF",command=lambda:camoff(),fg='white', bg='#75706f',width=20, height=5)
        btn2.grid(row=22, column=2)
        detect_signs(tk_win, label_widget_video)
    
    def camoff():
        label_widget_video.destroy()
        our_canvas2=Canvas(tk_win, width=8,height=8,bg='black').place(x=242,y=114)
        #btn2=Button(tk_win, text="WEBCAM ON",fg='white',bg='#75706f', command=run,width=20, height=5).place(x=690,y=540)
      
    def go_back():
        btn=Button(tk_win, text="KIDS MODE",command=lambda:go_on(), fg='white',bg='#75706f',width=20, height=5)
        btn.grid(row=3, column=3)
        mylabel1 = Label(tk_win,text='ASL Alphabet Recognition',font=('Helvetica', 26, 'bold'),bd=5,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE,width=43).place(x=200,y=20)
        label2=Label(tk_win,text='history:',font=('Helvetica', 16, 'bold'),bd=5,bg='#b4b4b4,',fg='#2c2c2c',relief=GROOVE,width=20)
        label2.grid(row = 1, column = 2,columnspan=3)

    def go_on():
        btn=Button(tk_win, text="BACK",command=lambda:go_back(), fg='white',bg='#75706f',width=20, height=5)
        btn.grid(row=3, column=4)
        mylabel1 = Label(tk_win,text='KIDS MODE',font=('Helvetica', 26, 'bold'),bd=5,bg='white',fg='#374254',relief=GROOVE,width=43)
        mylabel1.grid(row=3, column=3, columnspan=3)

        label5=Label(tk_win,text='images:',font=('Helvetica', 16, 'bold'),bd=5,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE,width=20)
        label5.grid(row = 1, column = 2)

    #btn1=Button(tk_win, text="EXIT",fg='white',bg='#75706f', command=tk_win.destroy,width=20, height=5)
    #btn1.grid(row=20, column=4, columnspan=1)
    btn=Button(tk_win, text="KIDS MODE",command=lambda:go_on(), fg='white',bg='#75706f',width=20, height=5)
    btn.grid(row=22, column=4, columnspan=2)
    btn2=Button(tk_win, text="WEBCAM ON",fg='white',bg='#75706f', command=run,width=20, height=5)
    btn2.grid(row=22, column=2, columnspan=2)
    
    run() # To make the video run without a button
    
   # Pack the Top Title on top-center of the window:







def detect_signs(tk_win: Tk,  label_widget_video: Label):
    
    letter = get_next_letter()
    
    def show_hint(letter):
        
        
    
    Button(win_tk, text="Hint", command = show_hint).pack(pady=10)
    
    #The labels, the letter that is recognised most will be on the top of the list in the interface
    def update_values():
        i = 2 ; a = 2
        for k,v in sorted(values.items(), key=lambda x: x[1], reverse=True):
            u = Label(tk_win,text=f'{k} : {v}',font=('Helvetica', 15, 'bold'),bg='white',fg='#374254',width=20)
            u.grid(row=i, column=a)
            i += 1
            if i == 20:
                i = 2 ; a += 2 #To create 2 columns 
    
    labels = {'0':'0','1': '1', '2': '2', '3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','a':'A','b':'B','c':'C','d':'D','e':'E','f':'F','g':'G','h':'H','i':'I','j':'J','k':'K','l':'L','m':'M','n':'N','o':'O','p':'P','q':'Q','r':'R','s':'S','t':'T','u':'U','v':'V','w':'W','x':'X','y':'Y','z':'Z'}
    
    # Load the pre-trained model
    our_model = joblib.load("model.joblib")
    
    # Init videocapture
    cap = cv2.VideoCapture(0)

    print("Camera is on... Entering the loop...")
    
    #For test reasons I added only 3 letters to the game
    lab = {'b':'Dataset_ASL/b/hand1_b_left_seg_1_cropped.jpeg','1': 'Dataset_ASL/1/hand1_1_bot_seg_2_cropped.jpeg', '4': 'Dataset_ASL/4/hand1_4_bot_seg_4_cropped.jpeg'}
    
    #To randomise the letters for game
    def getNextLetter(): 
        return random.choice(list(lab.keys()))
    letter = getNextLetter()
    
    #The dictionary for the scores
    data = pd.read_csv('dataset.csv')
    values = dict.fromkeys(set(data.label), 0)
    update_values()
        
    while cap.isOpened():
        _, img = cap.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # converting the channels
        coordinates, result = GetLandmarks(img)
        cv2.putText(img,  f"Show Letter {letter.upper()}",  (50, 100),  cv2.FONT_HERSHEY_SIMPLEX, 2,  (255, 255, 255),  6,  cv2.LINE_4) 
        print(values)

        if coordinates != '':
            img = cv2.cvtColor(DrawLandmarks(img, result), cv2.COLOR_RGB2BGR)
            coordinates = np.array(coordinates).reshape(1, 42)
            prediction = our_model.predict(coordinates)
            predicted_character = labels[(prediction[0])]
            
            if prediction == letter:
                color = (0,215,255)
                v = values.get(letter) + 1
                up_dict = {letter:v}
                values.update(up_dict)
                update_values()
                letter = getNextLetter()
            else :
                color = (0,0,0)     
            
            img = cv2.cvtColor(DrawBoundingBox(img, result, predicted_character,color), cv2.COLOR_RGB2BGR)
            print(predicted_character)

        img = cv2.resize(img, None, fx = 0.7, fy = 1.0)
        
        #For the hint image
        hint_image = cv2.resize(cv2.imread(lab[(letter[0])]), None, fx = 0.5, fy = 0.5)
        x_end = 690 + hint_image.shape[1]
        y_end = 0 + hint_image.shape[0]
        img[0:y_end,690:x_end] = cv2.cvtColor(hint_image, cv2.COLOR_RGB2BGR)
        
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

#tk_win.resizable(height = None, width = None)
tk_win.mainloop()

    