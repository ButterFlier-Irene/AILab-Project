from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('AILab-Project/coo.csv') #we have directly the file as a pandas dataframe

'''
y = np.array(data['labels'])
x = np.array(data['coordinates'])
x= x.reshape(2515, 1, 21)
#print(type(x))
print(x.shape)
print(y.shape)
#print(x.head())

#x_array = np.array(x)
'''

skf = StratifiedKFold(n_splits=5, shuffle=True)
# Cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(data.coordinates,data.labels)):
    X_train, X_test = np.array(data.coordinates[train_index]), np.array(data.coordinates[test_index])
    y_train, y_test = np.array(data.labels[train_index]), np.array(data.labels[test_index])
    #print(X_train,y_train)

# print(type(X_train))
# print(y_test.shape)  # we have 503 test elements
# print(y_train.shape) # we have 2012 train elements
'''
unique_elements = set(y_test)
count_unique = len(unique_elements)
print(count_unique)
'''

#OurRBFSVM =
