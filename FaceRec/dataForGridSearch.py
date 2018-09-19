import createHogOfAnImage
import numpy
def getDataForGridSearch ():
    hogs = createHogOfAnImage.hogGeneration('trainData', 2)
    labels = numpy.array([], dtype=int)
    for x in range(0, 200):
        labels = numpy.append(labels, 2)
    return hogs , labels
