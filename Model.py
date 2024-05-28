from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import ast 
from sklearn.decomposition import PCA as RandomizedPCA 
from sklearn.pipeline import make_pipeline
import seaborn as sns
# defining pca model
pca = RandomizedPCA(n_components=42, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced', C = 100, gamma = 0.01)
model = make_pipeline(pca, svc)

C_range = list(np.logspace(-2, 10, 13))
gamma_range = list(np.logspace(-9, 3, 13))
param_dist = dict(svc__C=C_range, svc__gamma=gamma_range)
grid = RandomizedSearchCV(model, param_dist, cv=skf, scoring='accuracy', n_iter=10)
#####################################################################################
data = pd.read_csv('coordinates.csv') #we have directly the file as a pandas dataframe
print(len(data.labels))
coo = np.array([np.array(co) for co in data.coordinates.apply(ast.literal_eval)]).reshape(2512, 42)
target = np.array(data.labels)


skf = StratifiedKFold(n_splits=5, shuffle=True)
accuracy_scores = []
######################################################################################
for fold, (train_index, test_index) in enumerate(skf.split(coo, target)):
    X_train_fold, X_test_fold = data.coordinates[train_index], data.coordinates[test_index]
    y_train_fold, y_test_fold = data.labels[train_index], data.labels[test_index]
    grid.fit(X_train_fold, y_train_fold)
    #we need the parameters of the best score from cv_results_
######################################################################################
'''
svc = SVC(kernel='rbf', class_weight='balanced', C = 100, gamma = 0.01)
model = make_pipeline(pca, svc)
model.fit(X_train, y_train)
# use the model to predict the test data, use it on new data
y_pred = model.predict(X_test)

# compute the accuracy: how good is the model
acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc) # 95% accuracy

pd.DataFrame(grid.cv_results_).to_csv('grid_search_results.csv')
svc = SVC(kernel='rbf', class_weight='balanced', C = 100, gamma = 0.01)
model = make_pipeline(pca, svc)
print(grid.best_score_, grid.best_params_)
print(grid.best_estimator_)
# WIP

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
                axi.imshow(X_test[i].reshape(62, 47), cmap='bone')
                axi.set(xticks=[], yticks=[])
                axi.set_ylabel(faces.target_names[y_pred[i]].split()[-1],
                        color='black' if y_pred[i] == y_test[i] else 'red') 
                
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);


# Clasification report

label = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# print(classification_report(y_test, y_pred, target_names = label))


# Confusion Matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                        xticklabels= label,
                        yticklabels = label)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()
'''


