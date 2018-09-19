import os
import cv2
import cropNsave
def renameAsIndexes(name):
    i = 0
    folder = r'/home/nvidia/Desktop/iSeeYou/' + name
    writeFolder = r'/home/nvidia/Desktop/FacesOfPeople/' + name
    if not os.path.exists(writeFolder):
        os.makedirs(writeFolder)
    imArray = cropNsave.load_images_from_folder(folder)
    for anImage in imArray:
        if anImage is not None:
            status = cv2.imwrite(os.path.join(writeFolder, str(i) + '.jpeg'), anImage)
            i = i + 1
    return


