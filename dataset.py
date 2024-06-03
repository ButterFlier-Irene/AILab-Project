import pandas as pd
import os
import numpy as np

'''
the purpose of this python file is to create a file 
containing the labels and the image name of the
Dataset_ASL folder.
The result is stored in the 'dataset.csv' file
Those images will be used later for creating a 
Dataset containing the labels with the hand coordinates
(see coo.py file).
'''
###############################################################################
#'label':['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','t','u','v','x','y','z'
#the number of labels should be 34 (10 numbers plus 24 letters of the alphabet)
###############################################################################

labels = []
images = []

for path in os.listdir('Dataset_ASL'):
    for i in os.listdir('Dataset_ASL/'+path):
        labels.append(path)                  # name of the image, contained in the list of directories.
        images.append(i)                     # labels (same as folder name).

d={'img':images,'label':labels}              # same number of images and labels.
df=pd.DataFrame(data=d)
df.to_csv('dataset.csv')                     # name of the file we are gonna store the result.
