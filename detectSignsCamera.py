from tkinter import GROOVE, Button, Frame, Label, Tk
from handLandmarks import GetLandmarks, DrawLandmarks, DrawBoundingBox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib, random
import pandas as pd
from kidsdictionary import get_kids_dict
from playsound import playsound
'''
Notice that when you are installing playsound, you need the version 1.2.2  .
Please, use the following command : pip install playsound==1.2.2  .
'''
'''
In this file we are using a GUI interface to show the model doing it's job.
If you are running this code on a Apple device, you can run directly the code below.
If you are on a Windows device you need to do a change before running the code:
    - change on line 43 the size of the font, from 26 to 17. This will help you
      to have a better experience with the interface
'''
def detect_image_gui(tk_win: Tk):
    '''
    We are s etting up the GUI interface by using  the
    tkinter functionalities.
    '''

    # Set the title of the main window
    tk_win.title('ASL Alphabet Recognition')
    tk_win.state('zoomed')
    tk_win.attributes('-fullscreen', True)
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
    title_label = Label(title_frame,text=' ASL Alphabet Recognition ',font=('Helvetica', 17, 'bold'),bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE) #put, instead of 26, if you are working on a windows computer
    title_label.grid(row = 0,column=3, columnspan=2,sticky='nsew')

    exit_button=Button(tk_win, text="EXIT",fg='black',bg='#75706f', command=tk_win.destroy,relief=GROOVE,height= int(height/250),padx=20)
    exit_button.grid(row=20, column=4,rowspan=2, sticky='nsew')

    def run_gamemode():
        '''
        Here we are defining the GAME MODE which will be the default mode
        '''
        kids_mode_button=Button(tk_win, text="KIDS MODE",command=lambda:run_kidsmode(),bd=3, fg='black',bg='#75706f',relief=GROOVE,height= int(height/250))
        kids_mode_button.grid(row=20, column=3,sticky='nsew',rowspan=2)
        detect_signs(tk_win, label_widget_video, kids_mode = False)

    def run_kidsmode():
        '''
        Here we are defining the changed interface
        for the KIDS MODE.
        '''
        kids_panel= Label(tk_win,text='Show a Sign',font=('Helvetica', 20, 'bold'),bd=3,bg='white',relief=GROOVE)
        kids_panel.grid(row = 2, column = 3,columnspan=2,rowspan=18, sticky='nsew')
        kids_mode_label=Label(tk_win,text='KIDS MODE',font=('Helvetica', 20, 'bold'),bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
        kids_mode_label.grid(row = 1, column = 3,columnspan=2,sticky='nsew')
        back_button=Button(tk_win, text="GAME MODE",command=lambda:run_gamemode(), fg='black',bg='#75706f',relief=GROOVE)
        back_button.grid(row=20, column=3,sticky='nsew',rowspan=2)
        
        detect_signs(tk_win, label_widget_video, kids_mode = True)

    run_gamemode() # To launch gamemode

###########################################################################################################


def detect_signs(tk_win: Tk,  label_widget_video: Label,kids_mode: bool):
    '''
    Here, with the first functions we are defining which image we have to show when
    we are in KIDS MODE. The aim is to show to the person,
    in this case a kid, the animal or the object associated
    to that specific letter he is showing.
    Afterwards we wrote the code to update and keep track
    the scores for each letter when in GAME MODE.
    '''
    # KIDS MODE part
    width = tk_win.winfo_screenwidth()
    height = tk_win.winfo_screenheight()
    score = 0
    
    imgs_dict = get_kids_dict() 
    
    # Function to display image given a specific label
    def open_img(letter,imgs_dict):
        '''
        Here we define the function to show the image
        associated to the letter the user is showing.
        And in the right position of the interface.
        '''
        image = imgs_dict[letter]
        # Convert the image from BGR (OpenCV format) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the image to a PIL format
        img_pil = Image.fromarray(image_rgb)
        img_pil = img_pil.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img_pil)
        panel = Label(tk_win, image=img_tk)
        panel.image = img_tk  # Keep a reference to avoid garbage collection
        panel.grid(row = 2, column = 3,columnspan=2,rowspan=18, sticky='nsew') #specify the position of the interface.


    show_hint_img = False
#########################################################################################################
    #GAME MODE part
    def show_hint():
        '''
        Here we define what the show hint once
        pressed should do.
        '''
        nonlocal show_hint_img
        if show_hint_img:
              show_hint_img = False
        else:
            show_hint_img = True
    
    #definition of dictionaries
    labels = {'0':'0','1': '1', '2': '2', '3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','a':'A','b':'B','c':'C','d':'D','e':'E','f':'F','g':'G','h':'H','i':'I','j':'J','k':'K','l':'L','m':'M','n':'N','o':'O','p':'P','q':'Q','r':'R','s':'S','t':'T','u':'U','v':'V','w':'W','x':'X','y':'Y','z':'Z'}
    lab = {'0':'Dataset_ASL/0/hand1_0_bot_seg_1_cropped.jpeg','1': 'Dataset_ASL/1/hand1_1_bot_seg_1_cropped.jpeg', '2': 'Dataset_ASL/2/hand1_2_bot_seg_1_cropped.jpeg', '3':'Dataset_ASL/3/hand1_3_bot_seg_1_cropped.jpeg','4':'Dataset_ASL/4/hand1_4_bot_seg_1_cropped.jpeg','5':'Dataset_ASL/5/hand1_5_bot_seg_1_cropped.jpeg','6':'Dataset_ASL/6/hand1_6_bot_seg_1_cropped.jpeg','7':'Dataset_ASL/7/hand1_7_bot_seg_1_cropped.jpeg','8':'Dataset_ASL/8/hand1_8_bot_seg_1_cropped.jpeg','9':'Dataset_ASL/9/hand1_9_bot_seg_1_cropped.jpeg','a':'Dataset_ASL/a/hand1_a_bot_seg_1_cropped.jpeg','b':'Dataset_ASL/b/hand1_b_left_seg_1_cropped.jpeg','c':'Dataset_ASL/c/hand1_c_bot_seg_1_cropped.jpeg','d':'Dataset_ASL/d/hand1_d_bot_seg_1_cropped.jpeg','e':'Dataset_ASL/e/hand1_e_bot_seg_1_cropped.jpeg','f':'Dataset_ASL/f/hand1_f_bot_seg_1_cropped.jpeg','g':'Dataset_ASL/g/hand1_g_bot_seg_1_cropped.jpeg','h':'Dataset_ASL/h/hand1_h_bot_seg_1_cropped.jpeg','i':'Dataset_ASL/i/hand1_i_bot_seg_1_cropped.jpeg','j':'Dataset_ASL/j/hand1_j_bot_seg_1_cropped.jpeg','k':'Dataset_ASL/k/hand1_k_bot_seg_1_cropped.jpeg','l':'Dataset_ASL/l/hand1_l_bot_seg_1_cropped.jpeg','m':'Dataset_ASL/m/hand1_m_bot_seg_1_cropped.jpeg','n':'Dataset_ASL/n/hand1_n_bot_seg_1_cropped.jpeg','o':'Dataset_ASL/o/hand1_o_bot_seg_1_cropped.jpeg','p':'Dataset_ASL/p/hand1_p_bot_seg_1_cropped.jpeg','q':'Dataset_ASL/q/hand1_q_bot_seg_1_cropped.jpeg','r':'Dataset_ASL/r/hand1_r_bot_seg_1_cropped.jpeg','s':'Dataset_ASL/s/hand1_s_bot_seg_1_cropped.jpeg','t':'Dataset_ASL/t/hand1_t_bot_seg_1_cropped.jpeg','u':'Dataset_ASL/u/hand1_u_bot_seg_1_cropped.jpeg','v':'Dataset_ASL/v/hand1_v_bot_seg_1_cropped.jpeg','w':'Dataset_ASL/w/hand1_w_bot_seg_1_cropped.jpeg','x':'Dataset_ASL/x/hand1_x_bot_seg_1_cropped.jpeg','y':'Dataset_ASL/y/hand1_y_bot_seg_1_cropped.jpeg','z':'Dataset_ASL/z/hand1_z_bot_seg_1_cropped.jpeg'}
    
    def update_values():
        '''
        Here we are setting the score, labels that
        will show how many times you have done the right 
        sign for each letter.
        '''
        label2=Label(tk_win,text=f' Score:  {score}' ,font=('Helvetica', 16, 'bold'),bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE)
        label2.grid(row = 1, column = 3,sticky='nsew')
        i = 2 ; a = 3
        #The labels, the letter that is recognised most will be on the top of the list in the interface
        for k,v in sorted(values.items(), key=lambda x: x[1], reverse=True):
            u =  Label(tk_win, text=f'{k.upper()}  =  {v}',font=('Helvetica', 16, 'bold'),bd=3,bg='white',fg='#374254',relief=GROOVE)
            u.grid(row=i, column=a,sticky='nsew')
            i += 1
            if i == 20:
                i = 2 ; a += 1 #To create 2 columns

    def getNextLetter(): 
        '''
        Here we are simply getting a random letter from
        a to z or a number between 1 and 9 for which the 
        user will have to show the correct sign.
        '''
        return random.choice(list(lab.keys()))
    letter = getNextLetter()
    
    #The dictionary for the scores
    if kids_mode == False:
        #The dictionary for the scores
        data = pd.read_csv('dataset.csv')
        values = dict.fromkeys(set(data.label), 0)
        update_values()
        hint_button = Button(tk_win, text="Hint", font=('Helvetica', 16, 'bold'), command = lambda:show_hint(),bd=3,bg='#b4b4b4',fg='#2c2c2c',relief=GROOVE, height=2)
        hint_button.grid(row = 1, column = 4 ,sticky='nsew')
        
    else:
        if letter.isalpha():
            open_img(letter,imgs_dict) 
    
    
#############################################################################################################
    # This part is dedicated to the use of the videocapture.

    # Load the pre-trained model
    our_model = joblib.load("model.joblib")
    
    # Init videocapture
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        _, img = cap.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # converting the channels
        coordinates, result = GetLandmarks(img)

        if coordinates != '':
            img = cv2.cvtColor(DrawLandmarks(img, result), cv2.COLOR_RGB2BGR)
            coordinates = np.array(coordinates).reshape(1, 42)
            prediction = our_model.predict(coordinates)
            predicted_character = labels[(prediction[0])]
            
            #if in kidmode, we want to check if the sign we are seeing is a letter. 
            #We don't have images corresponding to the numbers.
            if kids_mode == True:
                color = (0,0,0) 
                if prediction[0].isalpha():  #checking for aphabet letters since we have only images for these
                    open_img(prediction[0],imgs_dict)
            else:
                if prediction == letter:
                    color = (0,215,255)
                    playsound('Score.mp3')
                    v = values.get(letter) + 1
                    up_dict = {letter:v}
                    values.update(up_dict)
                    if show_hint_img: 
                        score -= 5
                        #show_hint_img = False
                    else:
                        score += 10
                    update_values()
                    letter = getNextLetter()
                else :
                    color = (0,0,0) 
                
            img = cv2.cvtColor(DrawBoundingBox(img, result, predicted_character,color), cv2.COLOR_RGB2BGR)
        
        img = cv2.resize(img, (int((width/4)*3), height), interpolation = cv2.INTER_LINEAR)
        
        #For the hint image
        if not kids_mode:
            cv2.putText(img,  f"Show Letter {letter.upper()}",  (50, 100),  cv2.FONT_HERSHEY_SIMPLEX, 2,  (255, 255, 255),  6,  cv2.LINE_4) 
            if show_hint_img == True :
                hint_image = cv2.resize(cv2.imread(lab[(letter[0])]), None, fx = 0.5, fy = 0.5)
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

if __name__ == "__main__":
    # Create the main window
    tk_win = Tk() 
    detect_image_gui(tk_win)


    