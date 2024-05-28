
#I used this official documentation of media pipe https://developers.google.com/mediapipe/api/solutions/js/tasks-vision.handlandmarker
#And the youtube videos that appear on google search, all of them that used mediapipe used the same method with different variable names.

# Also this google colab code which detected landmarks on images: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt

import cv2
import numpy as np
import joblib
from handLandmarks import GetLandmarks


labels = {'0':'0','1': '1', '2': '2', '3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','a':'A','b':'B','c':'C','d':'D','e':'E','f':'F','g':'G','h':'H','i':'I','j':'J','k':'K','l':'L','m':'M','n':'N','o':'O','p':'P','q':'Q','r':'R','s':'S','t':'T','u':'U','v':'V','w':'W','x':'X','y':'Y','z':'Z'}
#VideoCapture


OurModel = joblib.load("model.joblib")


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #converting the channels
    coordinates = GetLandmarks(image)
    #print(coordinates)
    if coordinates != []:
        np.array(coordinates).reshape(1, 42)
        prediction = OurModel.predict(coordinates)
    #print()
        predicted_character = labels[(prediction[0])]
        print(predicted_character)
    
    #prediction = model.predict([np.array(coordinates)])

    #predicted_character = labels_dict[int(prediction[0])]

    #print(predicted_character)





#I just tested also for the face for fun ;D, the code is basically the same
#    result2 = face.process(image)  #do face recognition on the frames from the videocapture camera
 #   if result2.multi_face_landmarks:
    #        for face_landmarks in result2.multi_face_landmarks:
       #         drawing_mp.draw_landmarks(
         #       img,
          #      face_landmarks,
           #     drawing_styles_mp.get_default_face_mesh_contours_style(),
                #connections=mp.solutions.face_mesh.FACEMESH_CONTOURS
             #   )    
##########

    cv2.imshow('Output',img)
    k = cv2.waitKey(10)

    if k == ord('q'):
        break

cap.release()  
cv2.destroyAllWindows()  
