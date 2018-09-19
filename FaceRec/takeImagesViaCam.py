import cv2
import os

def takeImagesViaCam(name):
    whereCamImagesWillBeSaved = '/home/nvidia/Desktop/ImageData/' + name
    camera = cv2.VideoCapture(1)
    if not os.path.exists(whereCamImagesWillBeSaved):
        os.makedirs(whereCamImagesWillBeSaved)
    imname = 0
    for i in range(0, 400):
        return_value, image = camera.read()
        status = cv2.imwrite(os.path.join(whereCamImagesWillBeSaved, str(imname) + '.jpeg'), image)
        imname = imname + 1
    del (camera)
