import cv2
import sys
import resize
import numpy

faceCascade = cv2.CascadeClassifier("/home/nvidia/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX
green = (0, 200, 0)


def realTimeFaceDetection(video_capture , name):

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = frame

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    img = frame
    # Draw a rectangle around the faces
    crop_img = numpy.zeros([600,600])
    fn = 0
    # faceNo = 0
    # croppedFaceArray = numpy.array([])
    for (x, y, w, h) in faces:
        #faceNo = faceNo + 1
        fn = fn + 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        crop_img = img[y:y + h, x:x + w]
        crop_img = img = resize.resize(crop_img)
        # imname = 'imno:' +str(imno)+ 'CroppedFace_no:' + str(faceNo)
        cv2.putText(frame, name, (x, y - 10), font, 1, green)
        # croppedFaceArray = numpy.append((croppedFaceArray , crop_img))

    #if faceNo>1:
        flag = 1
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if fn != 0: #and faceNo ==1:
        return crop_img #, flag
    if fn == 0:
        return None
    # if faceNo > 1:
    #     return croppedFaceArray,  flag


