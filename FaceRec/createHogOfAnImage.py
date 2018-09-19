import cv2
import numpy as np
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import resize

def hogGeneration(name , param):
    if param==2:
        personPath = r'/home/nvidia/Desktop/test/' + name
    else:
        personPath = r'/home/nvidia/Desktop/FacesOfPeople/' + name
    path, dirs, files = next(os.walk(personPath))
    hog = cv2.HOGDescriptor()
    #hm = hog matrice
    hm = np.array([])
    #the images after 50 shoots are considered because it takes some time for the camera to get used to the lighting of the shooting environment
    if param == 0:
        a = 1
        b = 40
    else:
        a = 50
        b = 250
    for x in range(a ,  b ):
        filename = personPath + '/' + str(x-1) + '.jpeg'
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        h = hog.compute(im, winStride=(600, 600), padding=(0, 0))  # storing HOG features as column vector
        h_trans = h.transpose()  # transposing the column vector
        if x == 50 and param != 0:
            hm = h_trans
        else:
            if x==1:
                hm = h_trans
            hm = np.vstack((hm, h_trans))  # appending it to the array
    #print 'hog submatrix is computed for ' + name + 'and it is in the size of ' + str(np.size(hm))
    return hm
