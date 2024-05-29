import os
import cv2
import pandas as pd
from handLandmarks import GetLandmarks
from torch.utils.data import Dataset
'''
The purpose of this python file is to create
our own Dataset, called in this case MpDataset
(Mp for Mediapipe) to get the images directories 
and apply a mediapipe transformation.
We are getting 21 points (42 coordinates) 
and their respective label.
On top of this we will build our machine learning
model (see 'Model.py' file)
The new dataset will appear in 'coo.csv' file'''
####################################################################
# Mediapipe contains mp.solutions, where we have the hands' landmarks.
# Since we are woking in static images, we'll need 'static_image_mode = True'
# The other parameters are
# - min_detection_confidence, to which we assign a minimum confidence
#   level when detecting hand
# - max_num_hands, that if setted to 1, will detect only one hand
# Since the images represent only one hand, we are sure that the 
# detected hand will be only one
####################################################################

class MpDataset(Dataset):
    '''we are creating a class by augmenting the Dataset
       class. We will get automatically the images from a
       given dataset containing paths and labels, we will 
       apply a mediapipe function to get the coordinates
    '''
    def __init__(self, imgs_dir = str, root = str):
        '''initialization:
        root: containing the root folder name
        imgs_dir: containing the name of the dataset
                  where name of images are stored
        labels: containing the dataset with labels
                and images
        coordinates: will contain the coordinates 
                     of the hand
        transform = will contain by default the 
                    mediapipe function hands.process # I removed this because now mediapipe functions are all in handLanmarks.py
        '''
        super().__init__()
        self.root = root
        self.labels = pd.read_csv(imgs_dir)
        self.coordinates=[]

    def len(self):
        '''will be the length of the labels'''
        return len(self.labels)
    
    def addBorder(self, img):
        '''add a black border to the images
        input the image you want to border
        and the output will be the bordered image
        '''
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
        return borderedimg

    def __getitem__(self, index):
        '''by recreating the image path
        we will read the image from the directory
        calculate the mediapipe coordinates
        and saving the coordinates into the respective
        variable
        '''
        img_path = os.path.join(self.root, self.labels.iloc[index, 2], self.labels.iloc[index, 1])
        
        img = cv2.imread(img_path)
        img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = self.labels.iloc[index, 2]
        result = GetLandmarks(imgRGB)[0]
        if result != '':                     #case where we detect the landmarks
            self.coordinates.append(result)     
        else:                                #case where we need to add the border
            imgRGB= self.addBorder(imgRGB)
            result = GetLandmarks(imgRGB)[0]
            self.coordinates.append(result)
           
####################################################################

data = MpDataset('dataset.csv', 'Dataset_ASL')

for c in data:
    '''we want to get all the images coordinates
    so to store them in the 'coo.csv' file'''
    pass

d = {'labels': data.labels['label'], 'coordinates': data.coordinates}

df=pd.DataFrame(data=d)
df.to_csv('coordinates.csv')
