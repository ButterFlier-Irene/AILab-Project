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
    X_train, X_test = data.coordinates[train_index], data.coordinates[test_index]
    y_train, y_test = np.array(data.labels[train_index]), np.array(data.labels[test_index])

X_train = np.array([np.array(co) for co in X_train.apply(ast.literal_eval)]).reshape(2010, 21, 2)
X_test = np.array([np.array(co) for co in X_test.apply(ast.literal_eval)]).reshape(502,21,2) #(no of total values, no of rows for each , no of columns) 
print(X_train.shape)
print(X_test.shape)



