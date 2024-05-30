import mediapipe as mp 
import cv2

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


def DrawBoundingBox(image, result, predicted_character):
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = image.shape  # Get image dimensions
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            # Convert normalized coordinates to pixel values
            min_x = int(min(x_coords) * w) - 10
            max_x = int(max(x_coords) * w) - 10
            min_y = int(min(y_coords) * h) - 10
            max_y = int(max(y_coords) * h) - 10
            # Draw rectangle on the image
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 0), 4)
            cv2.putText(image, 
                        predicted_character, 
                        (min_x, min_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.3, 
                        (0, 0, 0), 
                        3,
                        cv2.LINE_AA)
    return image
