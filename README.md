# FaceRecognitionByRE
Real time face recognition algorithm that uses opencv as cv2, numpy and sklearn as its mandatory libraries.

Recommended:
Ubuntu linux 16.04
python 2.x or 3.x

First you have to install python-pip
by pip or in linux: sudo apt-get install command you have to install numpy
opencv has to be installed on your device.
sklearn package can be installed similarly.

Usage:
create three folders named as FacesOfPeople , ImageData and ISeeYou
Then , open the "handler.py" class in the desired python-compiling environment
Run the handler class, for the first run, you have to input: data , name(s) , and in the 
grid search part you have to skip it because the first run has to import data. then you can train the classifier
with any dataset you want.

Important!
Do not give the same data that you trained your classifier as the input for the grid search-training 
The grid search takes 2-3 hours
I found the best resulting parameters by applying the grid search with a fast computing device(jetson tegra tx2)
The code will be further implemented for the "multiple face in a frame" case
