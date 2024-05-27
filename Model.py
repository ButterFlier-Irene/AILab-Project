from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
import ast 

data = pd.read_csv('coordinates.csv') #we have directly the file as a pandas dataframe


skf = StratifiedKFold(n_splits=5, shuffle=True)
# Cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(data.coordinates,data.labels)):
    X_train, X_test = data.coordinates[train_index].apply(ast.literal_eval), data.coordinates[test_index].apply(ast.literal_eval)
    y_train, y_test = np.array(data.labels[train_index]), np.array(data.labels[test_index])
#print(type(X_train[0]))
X_train = np.array([np.array(co) for co in X_train])
#print(len(X_train))
X_train1 = X_train.reshape(2010, 21, 2)
print(X_train1.shape)
#print(y_test.shape)
#print(len(X_train))
#X_train = X_train.reshape((2010, 21))
#X_test= X_test.reshape((2, 21)) 
#print(X_train.shape)

#print(X_test)
