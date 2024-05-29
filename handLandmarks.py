import mediapipe as mp 

landmarks = mp.solutions
hands_mp = landmarks.hands 
hands = hands_mp.Hands(static_image_mode = True, 
                       min_detection_confidence = 0.3,
                       max_num_hands = 1)

drawing_mp = landmarks.drawing_utils # module for drawing the landmarks on the screen
drawing_styles_mp = landmarks.drawing_styles # module for colored visualization of the landmarks #this helps us distinguish different fingers

def GetLandmarks(image):
    coordinates = '' 
    result = hands.process(image) 
    if result.multi_hand_landmarks:  #if many landmarks are detected
        for hand_landmarks in result.multi_hand_landmarks:   #for each hand landmark recognised 
            cooxy = []
            for dot in hand_landmarks.landmark:
                 cooxy += [dot.x,dot.y]
            coordinates = cooxy
    return coordinates,result

def DrawLandmarks(image, result):
    if result.multi_hand_landmarks:  #if many landmarks are detected
        for hand_landmarks in result.multi_hand_landmarks:   #for each hand landmark recognised 
            #this code is only for visualisation of the landmarks
            drawing_mp.draw_landmarks(
                image,  #image to draw the landmarks 
                hand_landmarks,  
                hands_mp.HAND_CONNECTIONS, #to draw the edges between landmarks
                drawing_styles_mp.get_default_hand_landmarks_style(),   #for colored landmarks
                drawing_styles_mp.get_default_hand_connections_style()  #for colored edges between landmarks
            )
    return image
