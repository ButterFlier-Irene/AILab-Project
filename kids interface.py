import mediapipe as mp
import cv2 
import joblib
import numpy as np
from handLandmarks import GetLandmarks

OurModel = joblib.load("model.joblib")
#in the dictonary each letter/digit must be connected to their img
d={'0':'0','1': '1', '2': '2', '3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','a':'A','b':'B','c':'C','d':'D','e':'E','f':'f.png','g':'G','h':'H','i':'I','j':'J','k':'K','l':'L','m':'M','n':'N','o':'O','p':'P','q':'Q','r':'R','s':'S','t':'T','u':'U','v':'V','w':'W','x':'X','y':'Y','z':'Z'}
cap = cv2.VideoCapture(0) 
while(cap.isOpened()): 
    ret, frame = cap.read() 
    coordinates = GetLandmarks(frame)
    if coordinates != []:
        np.array(coordinates).reshape(1, 42)
        y=OurModel.predict(coordinates)
        key=d.get(y)
    frame = cv2.resize(frame, (800, 600))
    logo = cv2.imread(key)
    size = 100
    logo = cv2.resize(logo, (size, size)) 
    img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY) 
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    roi = frame[-size-10:-10, -size-10:-10] 
    roi[np.where(mask)] = 0
    roi += logo 
    cv2.putText(frame, str(y), (5,80), 0.7,(255,255,255), 2)
    
    cv2.imshow("Hand Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release() 
cv2.destroyAllWindows() 

