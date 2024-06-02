
from tkinter import GROOVE, Button, Frame, Label, Tk, PhotoImage
from handLandmarks import GetLandmarks, DrawLandmarks, DrawBoundingBox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib, random
import pandas as pd
import IPython.display 
import time
#import playsound

def detect_image_gui(tk_win: Tk):
    
    # Set the title of the main window
    tk_win.title('ASL Alphabet Recognition')
    
    # Get the screen width and height
    width = tk_win.winfo_screenwidth()
    height = tk_win.winfo_screenheight()
    
    # Set the geometry of the main window to fill the entire screen
    tk_win.geometry("%dx%d" % (width, height))
    
    for i in range(0,21):
        tk_win.grid_rowconfigure(i, weight=1)

    # Create a frame that fills the entire window with a specific background color
    frame_video = Frame(tk_win, width=width, height=height,bg="#494848").place(x=0, y=0)
    label_widget_video = Label(frame_video)
    label_widget_video.grid(row = 0, column = 0, sticky = 'w',rowspan=22, columnspan=3)
    
    title_frame = Frame(tk_win, width=int(width/4), height=height,bg="#494848").place(x=int(width-(width/4)))
    title_label = Label(title_frame,text=' ASL Alphabet Recognition ',font=('Helvetica', 26, 'bold'),bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
    title_label.grid(row = 0,column=3, columnspan=2,sticky='nsew')

    exit_button=Button(tk_win, text="EXIT",fg='black',bg='#75706f', command=tk_win.destroy,relief=GROOVE,height= int(height/250),padx=20)
    exit_button.grid(row=20, column=4,rowspan=2, sticky='nsew')

    def run():
        kids_mode_button=Button(tk_win, text="KIDS MODE",command=lambda:go_on(), fg='black',bg='#75706f',relief=GROOVE,height= int(height/250))
        kids_mode_button.grid(row=20, column=3,sticky='nsew',rowspan=2)
        detect_signs(tk_win, label_widget_video, kids_mode = False)

    def go_on():
        img = PhotoImage()
        i = Label(tk_win, image= img,bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
        i.grid(row = 1, column = 3,columnspan=2,rowspan=19, sticky='nsew')
        kids_mode_label=Label(tk_win,text='KIDS MODE',font=('Helvetica', 20, 'bold'),bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
        kids_mode_label.grid(row = 1, column = 3,columnspan=2,sticky='nsew')
        back_button=Button(tk_win, text="GAME MODE",command=lambda:run(), fg='black',bg='#75706f',relief=GROOVE)
        back_button.grid(row=20, column=3,sticky='nsew',rowspan=2)
        
        detect_signs(tk_win, label_widget_video, kids_mode = True)

    run() # To make the video run without a button


def detect_signs(tk_win: Tk,  label_widget_video: Label,kids_mode: bool):
    width = tk_win.winfo_screenwidth()
    height = tk_win.winfo_screenheight()
    score = 0
    
    labels = {'0':'0','1': '1', '2': '2', '3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','a':'A','b':'B','c':'C','d':'D','e':'E','f':'F','g':'G','h':'H','i':'I','j':'J','k':'K','l':'L','m':'M','n':'N','o':'O','p':'P','q':'Q','r':'R','s':'S','t':'T','u':'U','v':'V','w':'W','x':'X','y':'Y','z':'Z'}
    lab = {'b':'Dataset_ASL/b/hand1_b_left_seg_1_cropped.jpeg','1': 'Dataset_ASL/1/hand1_1_bot_seg_2_cropped.jpeg', '4': 'Dataset_ASL/4/hand1_4_bot_seg_4_cropped.jpeg'}
    

    #The labels, the letter that is recognised most will be on the top of the list in the interface
    def update_values():
        label2=Label(tk_win,text=f' Score:  {score}' ,font=('Helvetica', 20, 'bold'),bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
        label2.grid(row = 1, column = 3,columnspan=2,sticky='nsew')
        i = 2 ; a = 3
        for k,v in sorted(values.items(), key=lambda x: x[1], reverse=True):
            u =  Label(tk_win, text=f'{k.upper()}  =  {v}',font=('Helvetica', 16, 'bold'),bd=3,bg='white',fg='#374254',relief=GROOVE)
            u.grid(row=i, column=a,sticky='nsew')
            i += 1
            if i == 20:
                i = 2 ; a += 1 #To create 2 columns 
                
    show_hint_img = False
    
    def show_hint():
        nonlocal show_hint_img
        if show_hint_img:
              show_hint_img = False
        else:
            show_hint_img = True
        
    

    #To randomise the letters for game
    def getNextLetter(): 
        return random.choice(list(lab.keys()))
    letter = getNextLetter()
    
    #The dictionary for the scores
    data = pd.read_csv('dataset.csv')
    values = dict.fromkeys(set(data.label), 0)
    if kids_mode == False:
        update_values()
    
    # Load the pre-trained model
    our_model = joblib.load("model.joblib")
    
    # Init videocapture
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        _, img = cap.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # converting the channels
        coordinates, result = GetLandmarks(img)
        cv2.putText(img,  f"Show Letter {letter.upper()}",  (50, 100),  cv2.FONT_HERSHEY_SIMPLEX, 2,  (255, 255, 255),  6,  cv2.LINE_4) 

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
                if kids_mode == False:
                    score += 10
                    update_values()
                letter = getNextLetter()
            else :
                color = (0,0,0)     
            
            img = cv2.cvtColor(DrawBoundingBox(img, result, predicted_character,color), cv2.COLOR_RGB2BGR)
        
        img = cv2.resize(img, (int((width/4)*3), height), interpolation = cv2.INTER_LINEAR)
        hint_button = Button(tk_win, text="Hint", command = show_hint,bd=3, fg='black',bg='#75706f', height=2).place(x =img.shape[1], y = 0)
        
        
        #For the hint image
        if show_hint_img == True:
            hint_image = cv2.resize(cv2.imread(lab[(letter[0])]), None, fx = 0.5, fy = 0.5)
            #x_end = 690 + hint_image.shape[1]  #890
            x_start = img.shape[1]-hint_image.shape[1] 
            y_end = 0 + hint_image.shape[0]
            img[0:y_end,x_start :img.shape[1]] = cv2.cvtColor(hint_image, cv2.COLOR_RGB2BGR)
        
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
tk_win.state('zoomed')
#tk_win.attributes('-fullscreen', True)
#tk_win.attributes('-zoomed', True)
tk_win.mainloop()

    