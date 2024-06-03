import mediapipe as mp 
import cv2

'''
Here we try to define the most important and basic functions that
Mediapipe can be useful for.
We defined three main functions:
    - get_landmarks, which gives us the coordinates of the hand
    - draw_landmarks, which draws on the input image the detected landmarks
    - draw_bounding_box, which highlights the hand with a rectangle  and shows the predicted value
'''

landmarks = mp.solutions
hands_mp = landmarks.hands 
hands = hands_mp.Hands(static_image_mode = True, 
                       min_detection_confidence = 0.3,
                       max_num_hands = 1)

drawing_mp = landmarks.drawing_utils         # module for drawing the landmarks on the screen
drawing_styles_mp = landmarks.drawing_styles # module for colored visualization of the landmarks #this helps us distinguish different fingers

def get_landmarks(image):
    
    '''
    With this function we want to extract the coordinates 
    from the input image and output, if there are, the coordinates
    in a list arranged form.
    '''
    coordinates = '' 
    result = hands.process(image) 
    if result.multi_hand_landmarks:                          #if many landmarks are detected
        for hand_landmarks in result.multi_hand_landmarks:   #for each hand landmark recognised 
            cooxy = []
            for dot in hand_landmarks.landmark:
                 cooxy += [dot.x,dot.y]
            coordinates = cooxy
    return coordinates,result

def draw_landmarks(image, result):
    '''
    With this function we want to draw the landmarks
    directly on the image we are inputting. 
    '''
    if result.multi_hand_landmarks:                                     #if many landmarks are detected
        for hand_landmarks in result.multi_hand_landmarks:              #for each hand landmark recognised 
            drawing_mp.draw_landmarks(
                image,                                                  #image to draw the landmarks 
                hand_landmarks,  
                hands_mp.HAND_CONNECTIONS,                              #to draw the edges between landmarks
                drawing_styles_mp.get_default_hand_landmarks_style(),   #for colored landmarks
                drawing_styles_mp.get_default_hand_connections_style()  #for colored edges between landmarks
            )
    return image


def draw_bounding_box(image, result, predicted_character,color):
    
    ''' 
    This function gets an image, the landmarks features
    and the predicted character and return the same image 
    with a rectangle indicating the hand detected with
    the predicted value.
    '''
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

            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color,4)
            cv2.putText(image, 
                        predicted_character, 
                        (min_x, min_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.3, 
                        color, 
                        3,
                        cv2.LINE_AA)
    return image
