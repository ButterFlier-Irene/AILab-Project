import cv2
import numpy as np
import joblib
from handLandmarks import GetLandmarks, DrawLandmarks

labels = {'0':'0','1': '1', '2': '2', '3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','a':'A','b':'B','c':'C','d':'D','e':'E','f':'F','g':'G','h':'H','i':'I','j':'J','k':'K','l':'L','m':'M','n':'N','o':'O','p':'P','q':'Q','r':'R','s':'S','t':'T','u':'U','v':'V','w':'W','x':'X','y':'Y','z':'Z'}
#VideoCapture

OurModel = joblib.load("model.joblib")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #converting the channels
    coordinates, result = GetLandmarks(image)

    if coordinates != '':
        img = cv2.cvtColor(DrawLandmarks(image,result), cv2.COLOR_RGB2BGR)
        coordinates = np.array(coordinates).reshape(1, 42)
        prediction = OurModel.predict(coordinates)
        predicted_character = labels[(prediction[0])]
        print(predicted_character)
    
    cv2.imshow('Detect Sign',img)
    k = cv2.waitKey(10)

    if k == ord('q'):
        break

cap.release()  
cv2.destroyAllWindows()  
