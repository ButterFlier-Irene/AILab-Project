# Letters and Numbers Sign Recognition (ASL)


### All the files contain detailed comments within them about the methods used.


Overview:

```
dataset.py - Creating a csv file with img paths
dataset.csv - The file created by dataset.py file
coordinates.py  - Creating the dataset using the previously created dataset.csv file
coordinates.csv - The dataset created by coordinates.py file which will be given to the model

handLandmarks.py - Contains all the media pipe functions
model.py - Creating the model
model.joblib - Our saved model 

kidsdictionary.py - Creating a dictionary for the kids imgs
detectSignsCamera.py - The interface and the videocapture. The main file that can be run

kidsimgs folder - Contains all the imgs for kids mode
Score.mp3 - Used for the gamemode
```