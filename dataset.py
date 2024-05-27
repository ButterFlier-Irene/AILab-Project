import pandas as pd
import os
import numpy as np
#'label':['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','t','u','v','x','y','z'
labels = []
images = []
#print(os.listdir('Dataset_ASL'))
for path in os.listdir('Dataset_ASL'):
    for i in os.listdir('Dataset_ASL/'+path):
        labels.append(path)
        images.append(i)
#print(len(labels))
#print(len(images))
d={'img':images,'label':labels}
df=pd.DataFrame(data=d)
df.to_csv('dataset.csv')