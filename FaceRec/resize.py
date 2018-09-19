import cv2

def resize(img):
    width = 600
    height = 600
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    return resized

