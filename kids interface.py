import mediapipe as mp
import cv2 
import joblib
import numpy as np
from handLandmarks import GetLandmarks
import detectSignsCamera

d={'0':'0','1': '1', '2': '2', '3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','a':'A','b':'B','c':'C','d':'D','e':'E','f':'f.png','g':'G','h':'H','i':'I','j':'J','k':'K','l':'L','m':'M','n':'N','o':'O','p':'P','Q':'q.png','r':'R','s':'S','t':'T','u':'U','v':'V','w':'W','x':'X','y':'Y','z':'Z'}
   
key=d[detectSignsCamera.predicted_character]
print(key)
'''
    cv2.imshow("Hand Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release() 
cv2.destroyAllWindows() 
'''
