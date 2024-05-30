from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
import ast 
from sklearn.decomposition import PCA as RandomizedPCA 
from sklearn.pipeline import make_pipeline
import seaborn as sns
import joblib

# defining pca model
#pca = RandomizedPCA(n_components=42, whiten=True, random_state=42)
#model = SVC(kernel = 'rbf', class_weight='balanced') #'C=1000000.0, gamma=0.001
#model = make_pipeline(pca, svc)
#####################################################################################
data = pd.read_csv('coordinates.csv') #we have directly the file as a pandas dataframe
coo = np.array([np.array(co) for co in data.coordinates.apply(ast.literal_eval)]).reshape(2512, 42)
target = np.array(data.labels)
#####################################################################################
X_train, X_test, y_train, y_test = train_test_split(coo, target, random_state=0, train_size=0.7)
#####################################################################################
skf = StratifiedKFold(n_splits=5, shuffle=True)

param_grid = {'C': list(np.logspace(-2, 10, 13)),  
              'gamma': list(np.logspace(-9, 3, 13)), 
              'kernel': ['rbf']} 
grid = RandomizedSearchCV(SVC(), param_grid, cv=skf, scoring='accuracy', n_iter=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_, grid.best_estimator_)

model = grid.best_estimator_

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(acc) # 95.22% accuracy

#joblib.dump(model, "model.joblib")
######################################################################################


'''
# Clasification report

label = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# print(classification_report(y_test, y_pred, target_names = label))


# Confusion Matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                        xticklabels= label,
                        yticklabels = label)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
'''
