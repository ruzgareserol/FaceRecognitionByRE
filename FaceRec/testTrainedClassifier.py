import cv2
from sklearn.externals import joblib
import numpy
import os
import cropNsave
import resize
from sklearn import svm, datasets
import numpy as np
from sklearn.metrics import accuracy_score
from ast import literal_eval



def predictionFunction(img , param , clf , namesArray  ):
    # crop faces and do some resizing and renaming

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog = cv2.HOGDescriptor()
    h = hog.compute(img, winStride=(600, 600), padding=(0, 0))
    h = h.transpose()


    if param == 0:
        return h
    if param == 1:
        prdct = clf.predict(h[0].reshape(1, -1))
        predictedName = str(namesArray[int(prdct) - 1])
        return 'this person is predicted to be '+str(predictedName)
    if param == 2:
        prdct = clf.predict(h[0].reshape(1, -1))
        return prdct






