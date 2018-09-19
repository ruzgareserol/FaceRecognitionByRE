import numpy as np
import cv2
import os
import resize
face_cascade = cv2.CascadeClassifier("/home/nvidia/opencv/data/haarcascades/haarcascade_frontalface_default.xml")

#takes images from faceFolder, crops faces and saves them into faceFolder
def cropAndSaveThemIntoSpecificFaceFolders ( param , name):
    if param == 0:
        imageFolder = imFolder = r'/home/nvidia/Desktop/ImageData/' + name
        imageArray = load_images_from_folder(imageFolder)
        imnum = 0
        for anImage in imageArray:
            cropFacesAndSaveThemIntoASpecificFolder(anImage, imnum, param, name)
            imnum = imnum + 1
    if param == 1:
        imageFolder = imFolder = r'/home/nvidia/Desktop/test/'
        cropFacesAndSaveThemIntoASpecificFolder(r'/home/nvidia/Desktop/test/test.jpeg', 1 , 1 , 'test')



#the name describes it all
def cropFacesAndSaveThemIntoASpecificFolder (img , imno, param, name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img", img)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faceNo = 0

    if param == 0:
        faceFolder =r'/home/nvidia/Desktop/iSeeYou/' + name
        if not os.path.exists(faceFolder):
            os.makedirs(faceFolder)
    if param == 1:
        faceFolder = r'/home/nvidia/Desktop/test/' + str(imno)
        if not os.path.exists(faceFolder):
            os.makedirs(faceFolder)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        crop_img = img[y:y + h, x:x + w]
        faceNo = faceNo + 1
        crop_img = img = resize.resize(crop_img)
        #imname = 'imno:' +str(imno)+ 'CroppedFace_no:' + str(faceNo)
        imname = str(imno)
        status = cv2.imwrite(os.path.join(faceFolder, imname + '.jpeg'), crop_img)
    return

#read images from a specific folder and encode them into the images[]
def load_images_from_folder(folder ):
    images = []
    h = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            h=h + 1
            images.append(img)
    return images
