
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from ROI_image_stream import *

################################################################
# Constants
################################################################
video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210813-182242_IL')
base_network = 'mobilenet'

################################################################
# Setup
################################################################
if base_network == 'mobilenet':
    base_network_inputsize = 224
elif base_network == 'inception_v3':
    base_network_inputsize = 300
else:
    raise(BaseException('Not implemented'))

istream = ROI_image_stream(video_path,ROI_size=base_network_inputsize)

istream.trainBackgroundSubtractor()

output_data = np.zeros((np.arange(1000, 54402, 5).shape[0], 4))
cumerror = 0
coor = []
# set for multiprocessing. reading frame automatically starts from this function
i = 0


framenum = np.array(np.arange(16330,30200,5))

try:
    chosen_image, coor = istream.getROIImage(framenum[i],previous_rc=coor)
    print(istream.backSub_lr)
    cv.imshow('Test', chosen_image)
    cv.imshow('Test2', istream.masked_image)
    cv.waitKey(100)
except BlobDetectionFailureError:
    cumerror += 1
    print(f'VideoProcessor : Couldn\'t find the ROI in Frame{framenum[i]}')
print(framenum[i])
i += 1



cv.imshow('Test2', istream.masked_image)
cv.waitKey(100)

istream.backSub.getBackgroundRatio()
istream.backSub.setHistory(100)
