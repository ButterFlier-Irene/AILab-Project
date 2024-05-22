import os
import mediapipe as mp
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
'''
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
        img = read_image(img_path)
        label = self.labels.iloc[index, 1]
        if self.transform:
            img =self.transform(img)
        return img, label
'''
#dictionary of labels
labels = {
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9',
    10:'a',
    11:'b',
    12:'c',
    13:'d',
    14:'e',
    15:'f',
    16:'g',
    17:'h',
    18:'i',
    19:'j',
    20:'k',
    21:'l',
    22:'m',
    23:'n',
    24:'o',
    25:'p',
    26:'q',
    27:'r',
    28:'s',
    29:'t',
    30:'u',
    31:'v',
    32:'w',
    33:'x',
    34:'y',
    35:'z'
}
#labels = [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','t','u','v','x','y','z']
#mp_dataset = MpDataset(labels, 'C:Dataset_ASL')
labels = pd.read_csv(labels)
