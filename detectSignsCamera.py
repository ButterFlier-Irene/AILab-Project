from tkinter import GROOVE, Button, Frame, Label, Tk
from handLandmarks import GetLandmarks, DrawLandmarks, DrawBoundingBox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib

def detect_image_gui(tk_win: Tk):
    
    # Set the title of the main window
    tk_win.title('ASL Alphabet Recognition')
    
    # Get the screen width and height
    width = tk_win.winfo_screenwidth()
    height = tk_win.winfo_screenheight()
    print("TKinter window size:", width, height)
    
    # Pack the Top Title on top-center of the window:
    Label(tk_win, text='ASL Alphabet Recognition', font=('Comic Sans MS', 24, 'bold'), bd=5, bg='#20262E', fg='#F5EAEA', relief=GROOVE).pack(pady=20, padx=20, anchor='n', side='top')
        
    # Set the geometry of the main window to fill the entire screen
    tk_win.geometry("%dx%d" % (width, height))
    
    # Create a frame that fills the entire window with a specific background color
    frame_1 = Frame(tk_win, width=width, height=height, bg="#181823").place(x=50, y=50)
    label_widget_video = Label(frame_1)
    label_widget_video.pack(pady=20, padx=20, anchor='s', side='bottom', expand=True)
    
    detect_signs(tk_win, label_widget_video)
    

def detect_signs(win_tk: Tk,  label_widget_video: Label):
    
    labels = {'0':'0','1': '1', '2': '2', '3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','a':'A','b':'B','c':'C','d':'D','e':'E','f':'F','g':'G','h':'H','i':'I','j':'J','k':'K','l':'L','m':'M','n':'N','o':'O','p':'P','q':'Q','r':'R','s':'S','t':'T','u':'U','v':'V','w':'W','x':'X','y':'Y','z':'Z'}
    
    # Load the pre-trained model
    our_model = joblib.load("model.joblib")
    
    # Init videocapture
    cap = cv2.VideoCapture(0)

    print("Camera is on... Entering the loop...")
    
    while cap.isOpened():
        _, img = cap.read()
        image_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # converting the channels
        
        coordinates, result = GetLandmarks(image_rgb)

        if coordinates != '':
            img = cv2.cvtColor(DrawLandmarks(image_rgb, result), cv2.COLOR_RGB2BGR)
            
            coordinates = np.array(coordinates).reshape(1, 42)
            prediction = our_model.predict(coordinates)
            predicted_character = labels[(prediction[0])]
            
            img = DrawBoundingBox(img, result, predicted_character)
            print(predicted_character)
        
        # Convert the image to a PIL image
        image_tk = Image.fromarray(img)
        
        # Convert the PIL image to a Tkinter PhotoImage
        final_tk_image = ImageTk.PhotoImage(image_tk)
        
        # Set the image in the TKinter label widget:
        label_widget_video.configure(image=final_tk_image)
        label_widget_video.image = final_tk_image
        
        # make TKinter window to refresh:
        win_tk.update()

    cap.release()



if __name__ == "__main__":
    # Create the main window
    tk_win = Tk()
    detect_image_gui(tk_win)
    
tk_win.mainloop()

    