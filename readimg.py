import os
import cv2
import pandas as pd
import mediapipe as mp
from torch.utils.data import Dataset

landmarks = mp.solutions #solutions module of mediapipe contains the ML models for body parts
hands_mp = landmarks.hands 
hands = hands_mp.Hands(static_image_mode = True , min_detection_confidence = 0.3)

class MpDataset(Dataset):
    #mandatory methods
    def __init__(self, labels_path, imgs_dir, transform = None):
        super().__init__()
        self.imgs_dir= imgs_dir
        self.transform=transform #storing transform to apply
        self.labels = pd.read_csv(labels_path)
        self.coordinates=[]
        #we are storing only labels, not images, since there could be a saturation problem

    def len(self):
        #since we have the label list we can simply return:
        return len(self.labels)
    
    def addBorder(img):
        #to add a border to imgs that landmarks are not detected
        border_color = [0,0,0]
        border_size = 248
        borderedimg = cv2.copyMakeBorder(
            img,
            top=border_size,
            bottom=border_size,
            left=80,
            right=80,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color
        )
        return borderedimg   #the input img to the process func

    def __getitem__(self, index):
        #we create the images path
        img_path = os.path.join(self.imgs_dir, self.labels.iloc[index, 2], self.labels.iloc[index, 1])
        #now we have a directory and a set of filenames. We want to put them together
        img = cv2.imread(img_path)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #we need to convert since otherwise mediapipe doesn't detect the landmarks.
        label = self.labels.iloc[index, 2]
        if self.transform:                             #here we will apply mediapipe
            result = self.transform(imgRGB)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    for dot in hand_landmarks.landmark:
                            xy = [dot.x,dot.y]
                            self.coordinates.append((xy))
        return imgRGB, self.coordinates, label        #will return a tuple containing (image, the LIST of coordinates, label of the image)

labels = pd.read_csv('dataset.csv')
data = MpDataset('dataset.csv', 'Dataset_ASL', hands.process)
img = data[100]
print(img[1]) #it will print the coordinates of mediapipe