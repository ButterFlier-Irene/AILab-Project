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
    def __init__(self, labels_path, imgs_dir):
        super().__init__()
        self.imgs_dir= imgs_dir
        self.transform= hands.process #storing transform to apply
        self.labels = pd.read_csv(labels_path)
        self.coordinates=[]
        #we are storing only labels, not images, since there could be a saturation problem

    def len(self):
        #since we have the label list we can simply return:
        return len(self.labels)
    
    def addBorder(self, img):
        #to add a border to imgs that landmarks are not detected
        border_color = [0,0,0]
        border_size = 248    #values that works best for our dataset
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

    def coo(self, img, result): #getting coordinates from hand
        coo = []
        for hand_landmarks in result.multi_hand_landmarks:
            for dot in hand_landmarks.landmark:
                    coo.append([dot.x,dot.y])

        self.coordinates.append(coo)

    def __getitem__(self, index):
        #we create the images path
        img_path = os.path.join(self.imgs_dir, self.labels.iloc[index, 2], self.labels.iloc[index, 1])
        #now we have a directory and a set of filenames. We want to put them together
        img = cv2.imread(img_path)
        img= cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        if img is None:
            print("Error: Unable to load the image.", img_path)
        else:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #we need to convert since otherwise mediapipe doesn't detect the landmarks.
            label = self.labels.iloc[index, 2]
            result = self.transform(imgRGB)
            if result.multi_hand_landmarks:
                self.coo(imgRGB, result)
            else:
                self.addBorder(imgRGB)                            #here we will apply mediapipe
                if result.multi_hand_landmarks:
                    self.coo(imgRGB, result)
            #return self.coordinates, label       #will return a tuple containing (image, the LIST of coordinates, label of the image)

labels = pd.read_csv('dataset.csv')
data = MpDataset('dataset.csv', 'Dataset_ASL')
for c in data:
    pass
d = {'labels': data.labels['label'], 'coordinates': data.coordinates}
#print(d)
print(len(d['labels']))
print(len(d['coordinates']))

'''
I want to save the coordinates into some kind of (possible numpy) array
What is the possible form?
label, coordinate [0, 0], coordinate[0,1]..., coordinate[n,0], coordinate[n,1]
'''

'''img = data[1]
print(len(img[0]))
img2 = data[2]
print(len(img2[0]))
img3=data[3]
print(len(img3[0]))
d={'label': [], 'coo': []}
for c in (img,img2, img3):
    if c!= None:
        print(len(c[0]))
        print(len(c[1]))
        d['label'] = c[1]
        d['coo'] = c[0]
#print(d)
#df=pd.DataFrame(data=d)
#df.to_csv('coo.csv')'''