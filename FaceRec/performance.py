import numpy as np
from sklearn.metrics import accuracy_score
import joblib
import obtainTestData
import dataForGridSearch

def performanceTest(clf, testFM , testLabels):
    #clf = joblib.load('/home/nvidia/PycharmProjects/FaceRec/FaceRecognizer.pk1')
    # testFM = obtainTestData.trainDataMatrix(clf)[0]
    # testLabels = obtainTestData.trainDataMatrix(clf)[1]
    # h = np.array([])
    predictScoreArray = np.array([], dtype=int)
    # for index in range(0 , testFM.__len__()):
    #     h = testFM[index]
    #     prdct = clf.predict(h.reshape(1, -1))
    #     predictScoreArray = np.append(predictScoreArray , prdct)
    # print predictScoreArray
    # print testLabels
    # accuracyScore = accuracy_score(predictScoreArray , testLabels)
    # print accuracyScore


    for index in range(0, testFM.__len__()):
        h = testFM[index]
        prdct = clf.predict(h.reshape(1, -1))
        predictScoreArray = np.append(predictScoreArray, prdct)
    accuracyScore = accuracy_score(predictScoreArray, testLabels)
    return accuracyScore