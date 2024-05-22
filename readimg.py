import os
import mediapipe as mp
from torch.utils.data import Dataset
import cv2
import pandas as pd

class MpDataset(Dataset):
    #mandatory methods
    def __init__(self, labels_path, imgs_dir, transform = None):
        super().__init__()
        self.imgs_dir= imgs_dir
        self.transform=transform #storing transform to apply
        self.labels = pd.read_csv(labels_path)
        #we are storing only labels, not images, since there could be a saturation problem

    def len(self):
        #since we have the label list we can simply return:
        return len(self.labels)

    def __getitem__(self, index):
        #we create the images path
        img_path = os.join(self.imgs_dir, self.labels.iloc[index, 0])#usually pandas reads by columns
        #now we have a directory and a set of filenames. We want to put them together
        img = cv2.imread(img_path)
        label = self.labels.iloc[index, 1]
        if self.transform:
            img =self.transform(img)
        return img, label
    

