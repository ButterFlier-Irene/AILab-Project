# Letters and Numbers Sign Recognition (ASL)


### All the Python files contain detailed comments within them about the methods used.

Files overview:

```
dataset.py - Creating a csv file with img paths
dataset.csv - The file created by dataset.py file (this file helps us with reading the imgs)
coordinates.py  - Creating the coordinate dataset using the previously created dataset.csv file
coordinates.csv - The csv file created by coordinates.py file which will be given to the model

handLandmarks.py - Contains all the MediaPipe functions
model.py - Creating the model
model.joblib - Our saved model 

kidsdictionary.py - Creating a dictionary for the kids imgs
detectSignsCamera.py - The interface and the videocapture. The main file that can be run

kidsimgs folder - Contains all the imgs for kids mode
Score.mp3 - Used for the gamemode
```
