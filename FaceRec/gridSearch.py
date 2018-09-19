import performance
import numpy
import sklearn
from sklearn.svm import SVC
import dataForGridSearch
import time

def gridSearch( FM , Labels):
    temp = 0
    ctemp = 0
    gtemp = 0
    testFM = dataForGridSearch.getDataForGridSearch()[0]
    testLabels = dataForGridSearch.getDataForGridSearch()[1]
    for g in range(1 , 101):
        for cc in range(1 , 11):
            start = time.time()
            totalElapsing = (g-1) + float(cc*0.2)
            gg = 0.001 * float(g)
            print 'grid search process ' + str(totalElapsing) + '%'
            gamma = gg
            print 'testing for gamma = ' + str(gamma) + 'and C= ' + str(cc)
            clf = SVC(gamma =gamma, C=cc , probability=True)
            clf.fit(FM, Labels)
            accuracyScore = performance.performanceTest(clf, testFM , testLabels)
            if accuracyScore>temp:
                temp = accuracyScore
                ctemp = cc
                gtemp = gamma
                print 'best accuracy score for now is ' + str(temp) + 'with gamma = ' + str(gtemp) + 'and C= ' + str(ctemp)
            end = time.time()
            timeElapsed = end - start
            remainingTime = (100 -totalElapsing)*timeElapsed
            print 'remaining time: ' + str(remainingTime / 60) + 'minutes'
    print 'grid search complete'
    print 'best accuracy score is ' + str(temp) + 'with gamma = ' + str(gtemp) + 'and C= ' + str(ctemp)
    print 'A classifier will be generated in consideration of these parameters'
    return temp , gtemp , ctemp