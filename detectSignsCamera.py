
from tkinter import GROOVE, Button, Frame, Label, Tk, Canvas,Listbox, PhotoImage
from handLandmarks import GetLandmarks, DrawLandmarks, DrawBoundingBox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import random
import pandas as pd
import time
#import IPython.display 
import tkinter as tk
import time
#import playsound

def detect_image_gui(tk_win: Tk):
    
    # Set the title of the main window
    tk_win.title('ASL Alphabet Recognition')
    
    # Get the screen width and height
    width = tk_win.winfo_screenwidth()
    height = tk_win.winfo_screenheight()
    #("TKinter window size:", width, height)
    
    
    # Set the geometry of the main window to fill the entire screen
    tk_win.geometry("%dx%d" % (width, height))
    
    for i in range(0,21):
    # Extend down to the bottom of the maximized window
        tk_win.grid_rowconfigure(i, weight=1)
        

    # Create a frame that fills the entire window with a specific background color
    frame_1 = Frame(tk_win, width=width, height=height,bg="#494848").place(x=0, y=0)
    label_widget_video = Label(frame_1)
    #frame_2 = Frame(tk_win, width=int(width/2), height=height).place(x=0, y=0)
    #label_widget_video = Label(frame_1, width=int(width/2))
    label_widget_video.grid(row = 0, column = 0, sticky = 'w',rowspan=22, columnspan=3)
    
    
    frame_2 = Frame(tk_win, width=int(width/4), height=height,bg="#494848").place(x=int(width-(width/4)))
    mylabel1 = Label(frame_2,text=' ASL Alphabet Recognition ',font=('Arial', 24, 'bold'),bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
    mylabel1.grid(row = 0,column=3, columnspan=2,sticky='nsew')
    
   # exit_button=Button(tk_win, text="     EXIT     ",fg='white',bg='#75706f', command=tk_win.destroy,relief=GROOVE, height= int(height/200))
   #exit_button=Button(tk_win, text="EXIT",fg='white',bg='#75706f', command=tk_win.destroy,relief=GROOVE, height= int(height/200),padx=20)
    exit_button=Button(tk_win, text="EXIT",fg='black',bg='#75706f', command=tk_win.destroy,relief=GROOVE,height= int(height/250),padx=20)
    exit_button.grid(row=20, column=4,rowspan=2, sticky='nsew')
    

    # Create a Listbox to display the history
    
    
    # Create a Listbox to display the webcam
    #label3=Label(tk_win, text='webcam:',font=('Helvetica', 16, 'bold'),bd=5,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE,width=18)
    #label3.grid(row = 1, column = 0,columnspan=2)


    #dataList = [4,5,6,3,7]
    #listbox = Listbox(tk_win, height=20) #right rectangle frame
    #àlistbox.grid(row = 2, column = 2, sticky = 'e', columnspan=2, rowspan=20)  


    # Add a line in canvas widget
    #our_canvas=Canvas(tk_win,width=1,height=1000,bg="white").grid(row = 0, column = 0, sticky = 'w')
   # our_canvas1=Canvas(tk_win,width=640,height=1,bg="white").place(x=640,y=500) 
    #our_canvas2=Canvas(tk_win, width=8,height=8,bg='black').place(x=242,y=114)

    def run():
        
        
        #kids_mode_button=Button(tk_win, text="KIDS MODE",command=lambda:go_on(), fg='white',bg='#75706f',relief=GROOVE, height= int(height/200))
        kids_mode_button=Button(tk_win, text="KIDS MODE",command=lambda:go_on(), fg='black',bg='#75706f',relief=GROOVE,height= int(height/250))
        kids_mode_button.grid(row=20, column=3,sticky='nsew',rowspan=2)
        
        detect_signs(tk_win, label_widget_video, kids_mode = False)
        
        #label3=Label(tk_win,text='Images:',font=('Helvetica', 16, 'bold'),bd=5,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
        #label3.grid(row = 1, column = 3,columnspan=2)
    
        #our_canvas2=Canvas(tk_win, width=8,height=8,bg='red').place(x=242,y=114)
        #btn2=Button(tk_win, text="WEBCAM OFF",command=lambda:camoff(),fg='white', bg='#75706f',width= butlab_w_size, height=butlab_h_size)
        #btn2.grid(row=20, column=2)
        
    
    def camoff():
        label_widget_video.destroy()
        #our_canvas2=Canvas(tk_win, width=8,height=8,bg='black').place(x=242,y=114)
        #btn2=Button(tk_win, text="WEBCAM ON",fg='white',bg='#75706f', command=run,width=20, height=5).place(x=690,y=540)
    
        

    def go_on():
        img = PhotoImage()
        i = Label(tk_win, image= img,bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
        i.grid(row = 1, column = 3,columnspan=2,rowspan=19, sticky='nsew')
        kids_mode_label=Label(tk_win,text='KIDS MODE',font=('Helvetica', 20, 'bold'),bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
        kids_mode_label.grid(row = 1, column = 3,columnspan=2,sticky='nsew')
        
        back_button=Button(tk_win, text="GAME MODE",command=lambda:run(), fg='black',bg='#75706f',relief=GROOVE)
        back_button.grid(row=20, column=3,sticky='nsew',rowspan=2)
        
        detect_signs(tk_win, label_widget_video, kids_mode = True)
        
       # label2=Label(tk_win,text='Images:',font=('Helvetica', 16, 'bold'),bd=5,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
       # label2.grid(row = 1, column = 3,columnspan=2)

    
    run() # To make the video run without a button
    
   # Pack the Top Title on top-center of the window:




# _tkinter.TclError: cannot use geometry manager pack inside . which already has slaves managed by grid






def detect_signs(tk_win: Tk,  label_widget_video: Label,kids_mode: bool):
    width = tk_win.winfo_screenwidth()
    height = tk_win.winfo_screenheight()
    score = 0

    
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
        
    
    labels = {'0':'0','1': '1', '2': '2', '3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','a':'A','b':'B','c':'C','d':'D','e':'E','f':'F','g':'G','h':'H','i':'I','j':'J','k':'K','l':'L','m':'M','n':'N','o':'O','p':'P','q':'Q','r':'R','s':'S','t':'T','u':'U','v':'V','w':'W','x':'X','y':'Y','z':'Z'}
    
    # Load the pre-trained model
    our_model = joblib.load("model.joblib")
    
    # Init videocapture
    cap = cv2.VideoCapture(0)

    print("Camera is on... Entering the loop...")
    
    #For test reasons I added only 3 letters to the game
    lab = {'b':'Dataset_ASL/b/hand1_b_left_seg_1_cropped.jpeg','1': 'Dataset_ASL/1/hand1_1_bot_seg_2_cropped.jpeg', '4': 'Dataset_ASL/4/hand1_4_bot_seg_4_cropped.jpeg', '0':'Dataset_ASL/0/hand1_0_bot_seg_5_cropped.jpeg', '2':'Dataset_ASL/2/hand1_2_bot_seg_1_cropped.jpeg', '3':'Dataset_ASL/3/hand1_3_bot_seg_3_cropped.jpeg', '5':'Dataset_ASL/5/hand1_5_dif_seg_3_cropped.jpeg', '6':'Dataset_ASL/6/hand1_6_dif_seg_5_cropped.jpeg', '7':'Dataset_ASL/7/hand1_7_right_seg_1_cropped.jpeg', '8':'Dataset_ASL/8/hand1_8_bot_seg_2_cropped.jpeg', '9':'Dataset_ASL/9/hand1_9_bot_seg_4_cropped.jpeg', 'a':'Dataset_ASL/a/hand1_a_bot_seg_3_cropped.jpeg', 'c':'Dataset_ASL/c/hand1_c_bot_seg_5_cropped.jpeg', 'd':'Dataset_ASL/d/hand1_d_bot_seg_3_cropped.jpeg', 'e':'Dataset_ASL/e/hand1_e_bot_seg_4_cropped.jpeg', 'f':'Dataset_ASL/f/hand1_f_left_seg_1_cropped.jpeg', 'g':'Dataset_ASL/g/hand1_g_bot_seg_4_cropped.jpeg', 'h':'Dataset_ASL/h/hand1_h_dif_seg_2_cropped.jpeg', 'i':'Dataset_ASL/i/hand1_i_bot_seg_3_cropped.jpeg', 'j':'Dataset_ASL/j/hand1_j_bot_seg_4_cropped.jpeg', 'k':'Dataset_ASL/k/hand1_k_bot_seg_4_cropped.jpeg', 'l':'Dataset_ASL/l/hand1_l_bot_seg_5_cropped.jpeg', 'm':'Dataset_ASL/m/hand1_m_bot_seg_1_cropped.jpeg', 'n':'Dataset_ASL/n/hand1_n_bot_seg_2_cropped.jpeg', 'o':'Dataset_ASL/o/hand1_o_bot_seg_3_cropped.jpeg', 'p':'Dataset_ASL/p/hand1_p_bot_seg_4_cropped.jpeg', 'q':'Dataset_ASL/q/hand1_q_right_seg_2_cropped.jpeg', 'r':'Dataset_ASL/r/hand1_r_bot_seg_1_cropped.jpeg', 's':'Dataset_ASL/s/hand1_s_bot_seg_2_cropped.jpeg', 't':'Dataset_ASL/t/hand1_t_bot_seg_3_cropped.jpeg', 'u':'Dataset_ASL/u/hand1_u_bot_seg_4_cropped.jpeg', 'v':'Dataset_ASL/v/hand1_v_bot_seg_5_cropped.jpeg', 'w':'Dataset_ASL/w/hand1_w_bot_seg_1_cropped.jpeg', 'x':'Dataset_ASL/x/hand1_x_bot_seg_2_cropped.jpeg', 'y':'Dataset_ASL/y/hand1_y_bot_seg_3_cropped.jpeg', 'z':'Dataset_ASL/z/hand1_z_bot_seg_4_cropped.jpeg'}
    
    #To randomise the letters for game
    def getNextLetter(): 
        return random.choice(list(lab.keys()))
    letter = getNextLetter()
    
    
    #The dictionary for the scores
    data = pd.read_csv('dataset.csv')
    values = dict.fromkeys(set(data.label), 0)
    if kids_mode == False:
        update_values()
        
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
        hint_button = Button(tk_win, text="Hint", command = show_hint,bd=3, fg='black',bg='#75706f', height=2).place(x =img.shape[-1], y = 0)
        #hint_button.grid(row = 1, column = 3,columnspan=2,sticky='nsew')
        
        
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

    