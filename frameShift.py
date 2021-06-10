import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

def train_backsub(vid, backSub, stride=2000):
    """
    train_backsub : train BackgroundSubtractorKNN
    ---------------------------------------------------------------- 
    vid : cv.VideoCapture object : Target video
    backSub : cv.BackgroundSubtractorKNN object : Target BS object
    stride : int : Frame stride for training purpose 
    """
    count = 0
    tic = time.time()
    #TRAIN
    while True:
        vid.set(cv.CAP_PROP_POS_FRAMES, count)
        ret, image = vid.read()
        if image is None:
            print("End of the File")
            toc = time.time()
            spent_time = (toc - tic)
            print(spent_time)
            break
        else:
            masked_image = backSub.apply(image)
            print(str(count) + 'th image is used for the training')
            count += stride
            continue

def returnROI(vid, backSub, frame_number, ROI_size = 150):
    """
    returnROI : return ROI frame from the video
    ----------------------------------------------------------------
    vid : cv.VideoCapture object : Target video
    backSub : cv.BackgroundSubtracorKNN object : trained BS object
    frame_number : int or int numpy array : frame to process
    ROI_size : int : ROI size (square)
    ---------------------------------------------------------------- 
    outputFrame : 3D numpy array : processed frame(s)  
    center : 2D numpy array : exact location of the center of the frame
    """
    vid.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, image = vid.read()
    masked_image = backSub.apply(image)
    plt.imshow(masked_image)
    image = cv.copyMakeBorder(image, ROI_size, ROI_size, ROI_size, ROI_size, cv.BORDER_CONSTANT, value=[0, 0, 0])

    # SimpleBlobDetector Setup
    parameter = cv.SimpleBlobDetector_Params()
    parameter.filterByArea = True
    parameter.minArea = 2000
    parameter.filterByColor = True
    parameter.blobColor = 255
    parameter.filterByInertia = False
    parameter.filterByConvexity = False
    detector = cv.SimpleBlobDetector_create(parameter)
    detected_blob = detector.detect(masked_image)
    
    if len(detected_blob) != 0:
        blob_centerY, blob_centerX = int(detected_blob[0].pt[1]+ROI_size), int(detected_blob[0].pt[0]+ROI_size)
        chosen_image = image[blob_centerY-ROI_size:blob_centerY+ROI_size, blob_centerX-ROI_size:blob_centerX+ROI_size]
    else:
        print(len(detected_blob))
        raise("multiple blob detected!")
    return (chosen_image, [blob_centerY, blob_centerX])

ROI = 150
frame_number = 6324
#chosen_image = taking_rat_photo(vid, ROI, frame_number)
path_video = Path('/mnt/Data/Data/Small Experiment/LARGE/2021-04-22 11-47-11.mkv')
vid = cv.VideoCapture(str(path_video))
backSub = cv.createBackgroundSubtractorKNN()
train_backsub(vid, backSub)


for i in [4423]:
    print("showing %d frame" % i)
    return_image = returnROI(vid,backSub,i)
    plt.imshow(return_image[0])


