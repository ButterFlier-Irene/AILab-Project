import mediapipe as mp 

landmarks = mp.solutions #solutions module of mediapipe contains the ML models for body parts
hands_mp = landmarks.hands   #get hands from mp solutions

drawing_mp = landmarks.drawing_utils # module for drawing the landmarks/dots on the screen
drawing_styles_mp = landmarks.drawing_styles # module for colored visualization of the landmarks #this helps us distinguish different fingers

hands = hands_mp.Hands(max_num_hands = 1, static_image_mode = True , min_detection_confidence = 0.5)

def GetLandmarks(image):
    coordinates = [] #empty array to store coordinates of landmarks 
    result = hands.process(image) 
    if result.multi_hand_landmarks:  #if many landmarks are detected
        for hand_landmarks in result.multi_hand_landmarks:   #for each hand landmark recognised 

            #this code is only for visualisation of the landmarks
            drawing_mp.draw_landmarks(
                image,
                hand_landmarks,  #to draw the dots/the landmarks
                hands_mp.HAND_CONNECTIONS, #to draw the edges between landmarks
                drawing_styles_mp.get_default_hand_landmarks_style(),   #for colored landmarks
                drawing_styles_mp.get_default_hand_connections_style()  #for colored edges between landmarks
            )
            coo = []
            #this code is where the data is taken from the landmarks    

            # Each landmark coordinates x and y (also z) can be obtained by : 
            # iterating through all the hand landmarks and putting in a numpy array
            # later we will give this numpy array to the model to predict.
            for dot in hand_landmarks.landmark:
                 
                 coo += [dot.x,dot.y]
            coordinates.append(coo)
    return coordinates