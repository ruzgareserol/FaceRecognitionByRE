import createHogOfAnImage
import resize
import cv2
import sklearn
from sklearn import grid_search , cross_validation
import numpy as np

def trainDataMatrix(clf):
    names = np.array([], dtype=str)
    testData = np.array([])
    # take names from database
    nameFolder = open('/home/nvidia/Desktop/ImageData/dataBase/names', 'r')

    namesArray = nameFolder.read().split("' '")
    nameFolder.close()
    eNum = 0
    for element in namesArray:
        newElement = ""
        for char in range(0, len(element)):
            if element[char] == '[' or element[char] == ']' or element[char] == "'":
                newElement = newElement
            else:
                newElement = newElement + element[char]
        names = np.append(names, newElement)
        eNum = eNum + 1

    trainHogs = np.array([])
    trainFM = np.array([])

    x = 0
    for name in names:
        trainHogs = createHogOfAnImage.hogGeneration(str(name), 0)
        if x ==0:
            trainFM = trainHogs
            x = 1
        else:
            trainFM = np.vstack((trainFM , trainHogs ))


    labels = np.array([], dtype=int)
    pplcount = np.size(names)
    giveLabel = 1
    for x in range(0, pplcount):
        for i in range(0, 40):
            labels = np.append(labels, giveLabel)
        giveLabel = giveLabel + 1
    #clf.fit(trainFM , labels)
    return trainFM , labels