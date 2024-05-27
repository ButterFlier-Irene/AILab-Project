from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('coo.csv') #we have directly the file as a pandas dataframe

skf = StratifiedKFold(n_splits=5, shuffle=True)
# Cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(data.coordinates,data.labels)):
    X_train, X_test = np.array(data.coordinates[train_index]), data.coordinates[test_index]
    y_train, y_test = np.array(data.labels[train_index]), np.array(data.labels[test_index])
print(type(X_train[0]))
#print(y_train.shape)
#print(y_test.shape)
'''
X_train = np.ndarray((2012, 21, 2), X_train)
X_test = np.ndarray((502, 21, 2), X_test)
print(X_train.shape)
print(X_test.shape)
print(X_test)'''