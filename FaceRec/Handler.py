#this class, as can be seen from it's name is the handler of the whole project
#it handles the packages, naming, image processing, cropping and even training the svm classifier

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import numpy
import os
import glob
import cv2
import sys
import takeImagesViaCam
import createHogOfAnImage
import resize
import renameFilesAsIndexes
import cropNsave
import time
import testTrainedClassifier
import realTime
import gridSearch

print 'welcome to FaceRecognizer'
#first of all, we have to take images via cam.
pplCount = 0
labels = numpy.array([])
names = numpy.array([])
nameLabels = numpy.array([], dtype=int)
print 'gathering and processing data from database'

for filename in os.listdir('/home/nvidia/Desktop/FacesOfPeople/'):
    pplCount = pplCount+1
    nameLabels = numpy.append(nameLabels , int(pplCount))
    names = numpy.append(names , str(filename))
    HM = createHogOfAnImage.hogGeneration(filename , 1)
    if pplCount <= 1:
        FM = HM
    else:
        FM = numpy.vstack((FM, HM))
    for x in range(0, 200):
        labels = numpy.append(labels, int(pplCount))


print'Names and corresponding labels are:'
print (names)
print (nameLabels)
print 'currently we have ' + str(pplCount) + ' people'
#print'HOG matrice:'
#print(HM)






decision = raw_input('to update database type "data"')

if decision == 'data':
    while True:
        print 'you can exit the input stage if you input "done"  '
        name = raw_input('input a person name =')
        print 'People count is ' + str(pplCount)
        if name == 'done':
            print 'finished importing people.'
            break
        else:
            pplCount = pplCount + 1
            names = numpy.append(names, str(name))
            takeImagesViaCam.takeImagesViaCam(name)
            print 'finished capturing'
            # from this point, we have images of people, saved in imageData/theirName
            # now, we are going to crop their faces via haar_cascade of cv2
            # the cropped images will be saved in a seperate folder called iSeeYou/theirName
            # they will also be resized to 450x450 images
            cropNsave.cropAndSaveThemIntoSpecificFaceFolders(0, name)
            print 'finished cropping and saving in the folder: iSeeYou'
            renameFilesAsIndexes.renameAsIndexes(name)
            print 'images are now renamed as indexes and saving in the folder: facesOfPeople'
            # from this point, we have faces cropped and resized for our purpose of HOG generation
            # imageData contains cam outputs
            # iSeeYou contains cropped-resized images
            # facesOfPeople contains the data we need for HOG generation
            # let's begin generating HOGs for multiclass svm classification and prediction
            HM = createHogOfAnImage.hogGeneration(name , 1)
            print 'succesfully created a feature submatrix for ' + name
            if pplCount <= 1:
                FM = HM

            else:
                FM = numpy.vstack((FM, HM))
            print name + 'is added to feature matrix'
            nameLabels = numpy.append(nameLabels, pplCount)
            print 'feature submatrix:'
            print (FM)
            for x in range(0, 200):
                labels = numpy.append(labels, int(pplCount))

    # let's complete the construction of the classifier
    # from now the classifier will be trained
    print FM
    print labels

    print '----------------------------------------'
    print'Names and corresponding labels are'
    print (names)
    print (nameLabels)
    print 'currently we have ' + str(pplCount) + ' people'

    file2write = open('/home/nvidia/Desktop/ImageData/dataBase/names', 'w')
    file2write.write(str(names))
    file2write.close()

    file2write2 = open('/home/nvidia/Desktop/ImageData/dataBase/nameLabels', 'w')
    file2write2.write(str(nameLabels))
    file2write2.close()
    print 'database updated'
    # finished constructing the featres matrix and labels
    # now it is time to construct the classifier

    print 'constructing the classifier'
    decision = raw_input('to start grid search, type "grid"')
    if decision =="grid":
        gridsearch = gridSearch.gridSearch(FM , labels )
        g = gridsearch[1]
        cc = gridsearch[2]
        print 'parametrisation complete'
    else:
        g =0.098
        cc = 1
        print 'gamma = 0.098 and C = 1 is for the classifier'
    clf = SVC(gamma=g, C=cc, probability=True)
    clf.fit(FM, labels)
    print 'Feature matrix and Labels are succesfully fitted into the classifier'
    joblib.dump(clf , "FaceRecognizer.pk1" ,compress= 4)
    print 'classifier is saved as FaceRecognizer.pk1'
else:
    print'skipping database implementation'
print 'loading classifier'
clf = joblib.load('/home/nvidia/PycharmProjects/FaceRec/FaceRecognizer.pk1')
print 'Classifier ready'
prob = numpy.array([] , dtype = float)
realTimeLounch = raw_input('for real time face recognition input: begin')
print 'you can quit real time face recognition by pressing the "q" key'
if realTimeLounch == 'begin':
    video_capture = cv2.VideoCapture(1)
    tempName = ""
    idkCount = 0
    iKnowCount = 0
    prdctArray = numpy.array([] , dtype=int)
    while True:
        face = realTime.realTimeFaceDetection(video_capture , tempName)#[0]
        # flag = realTime.realTimeFaceDetection(video_capture , tempName)[1]
        if numpy.size(face)>3: #and flag ==0:
            hh = testTrainedClassifier.predictionFunction(face, 0, clf , names)
            prob = (clf.predict_proba(hh))
            prdct = testTrainedClassifier.predictionFunction(face, 2, clf, names)
            if (numpy.amax(prob))>0.5:

                predictedName = str(names[int(prdct) - 1])
                tempName = predictedName
                print numpy.amax(prob)
                idkCount = 0
            else:
                idkCount = idkCount + 1
            if idkCount>60:
                tempName = "unknown"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print 'no face detected'

        # if numpy.size(face) > 3 and flag> 0:
        #     for f in face:
        #         hh = testTrainedClassifier.predictionFunction(f, 0, clf, names)
        #         prob = (clf.predict_proba(hh))
        #         prdctArray = numpy.append(testTrainedClassifier.predictionFunction(f, 2, clf, names))
        #         if (numpy.amax(prob)) > 0.5:
        #             predictedName = str(names[int(prdct) - 1])
        #             tempName = predictedName
        #             print numpy.amax(prob)
        #             idkCount = 0
        #         else:
        #             idkCount = idkCount + 1
        #         if idkCount > 60:
        #             tempName = "unknown"
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # else:
        #     print 'no face detected'

    video_capture.release()
    cv2.destroyAllWindows()








