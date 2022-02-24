"""
istream_test.py
2021 Knowblesse
Testing script for Dilation & Erosion parameter optimization
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from ROI_image_stream import ROI_image_stream

################################################################
# Constants
################################################################
video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210629-183643_PL')
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

DEBUG_FRAME = 40555 # line + artifact
istream.vid.set(cv.CAP_PROP_POS_FRAMES,DEBUG_FRAME)
ret, image = istream.vid.read()

fig1 = plt.figure(1)
fig1.clf()
ax1 = fig1.subplots(1,2)
ax1[0].imshow(image)
ax1[0].set_title('Original')

image = cv.threshold(image, 120, 0, cv.THRESH_TOZERO)[1]

ax1[1].imshow(image)
ax1[1].set_title('Threshold')

masked_image = istream.backSub.apply(image,learningRate = 1e-6) # according to the document, 0 should freeze the backgroundsubtractor, but it fails. So we put very small number
image = cv.copyMakeBorder(image, istream.half_ROI_size, istream.half_ROI_size, istream.half_ROI_size, istream.half_ROI_size, cv.BORDER_CONSTANT, value=[0, 0, 0])

fig2 = plt.figure(2)
fig2.clf()
ax2 = fig2.subplots(2,3)
ax2[0,0].imshow(image)
ax2[0,0].set_title('Original')

def dilate(image, size):
    size = int(size)
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size,size),((int((size-1)/2), int((size-1)/2))))
    return cv.dilate(image, element)

def erode(image, size):
    size = int(size)
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size,size),((int((size-1)/2), int((size-1)/2))))
    return cv.erode(image, element)

def getKernel(size):
    size = int(size)
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size,size),((int((size-1)/2), int((size-1)/2))))

#         denoised_mask = dilate(erode(dilate(erode(masked_image, 3), 4), 8), 6)
denoised_mask = masked_image
denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, getKernel(3))
ax2[0,1].imshow(denoised_mask)
ax2[0,1].set_title('Open1')

denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, getKernel(5))
ax2[0,2].imshow(denoised_mask)
ax2[0,2].set_title('Open2')

denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_CLOSE, getKernel(10))
ax2[1,0].imshow(denoised_mask)
ax2[1,0].set_title('Open3')

denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, getKernel(7))
ax2[1,1].imshow(denoised_mask)
ax2[1,1].set_title('Open4')

denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, getKernel(12))
ax2[1,2].imshow(denoised_mask)
ax2[1,2].set_title('Open5')

useSimpleBlobDetector = False

if useSimpleBlobDetector:
    parameter = cv.SimpleBlobDetector_Params()
    parameter.filterByArea = True
    parameter.filterByConvexity = True
    parameter.filterByCircularity = True
    parameter.filterByInertia = False
    parameter.filterByColor = False
    parameter.minArea = 500  # this value defines the minimum size of the blob
    parameter.maxArea = 10000  # this value defines the maximum size of the blob
    parameter.minDistBetweenBlobs = 1
    parameter.minConvexity = 0.3
    parameter.minCircularity = 0.3

    detector = cv.SimpleBlobDetector_create(parameter)
    detected_blob = detector.detect(denoised_mask)
    max_blob_index = 0
    if len(detected_blob) > 1:# if multiple blob is detected, select the largest one
        max_blob_size = 0
        for i, blob in enumerate(detected_blob):
            print(blob.size)
            print(blob.pt)
            if max_blob_size < blob.size:
                max_blob_size = blob.size
                max_blob_index = i
        print('Multiple blobs : %d detected' % len(detected_blob))
        showChosenImage = True
    blob_center_row, blob_center_col = int(np.round(detected_blob[max_blob_index].pt[1])) , int(np.round(detected_blob[max_blob_index].pt[0]))
else:
    # Custom BlobDetector
    """
    """
    minBlobSize = 1000
    maxLength = 200

    cnts = cv.findContours(denoised_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    blob_idx = [i for i in range(len(cnts))]
    blob_area = np.array([cv.contourArea(i) for i in cnts])
    blob_rect = [cv.minAreaRect(i) for i in cnts]
    blob_rect_size = np.array([rect[1] for rect in blob_rect])

    for idx in blob_idx:
        # 1. exclude blobs with small size
        if blob_area[idx] < minBlobSize: # too small
            blob_idx.remove(idx)
            continue
        # 2. exclude blobs with long size
        if np.all(blob_rect_size[idx][1] > maxLength): # too long
            blob_idx.remove(idx)
            continue
    # 3. if still multiple blobs exists, select the most round one
    if len(blob_idx) > 1:
        blob_not_circularity = np.squeeze(np.abs(np.diff(blob_rect_size))) / (blob_area ** 0.5)
        blob_idx = [blob_idx[np.argmin(blob_not_circularity[blob_idx])]]
    blob_center_col, blob_center_row = list(map(int, blob_rect[blob_idx[0]][0]))

chosen_image = image[
               blob_center_row - istream.half_ROI_size + istream.half_ROI_size : blob_center_row + istream.half_ROI_size + istream.half_ROI_size,
               blob_center_col - istream.half_ROI_size + istream.half_ROI_size : blob_center_col + istream.half_ROI_size + istream.half_ROI_size,:]

plt.figure(2)
plt.clf()
plt.imshow(chosen_image)
