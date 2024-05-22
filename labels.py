import pandas as pd

labels = { 'label':[0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','t','u','v','x','y','z']}

dataset = pd.DataFrame(labels)
print(dataset)
dataset.to_csv('labels.csv')