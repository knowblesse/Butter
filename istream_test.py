"""
istream_test.py
2021 Knowblesse
Testing script for Dilation & Erosion parameter optimization
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from helper_function import ROI_image_stream

################################################################
# Constants
################################################################
path_dataset = Path.home() / 'VCF/butter/dataset/Ex1/0422_02_Valium_0.2/'
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

istream = ROI_image_stream(path_dataset,ROI_size=base_network_inputsize)
istream.trainBackgroundSubtractor(stride=1000)


DEBUG_FRAME = 1060

istream.vid.set(cv.CAP_PROP_POS_FRAMES,DEBUG_FRAME)
ret, image = istream.vid.read()

image = cv.threshold(image, 80, 0, cv.THRESH_TOZERO)[1]
masked_image = istream.backSub.apply(image,learningRate = 1e-6) # according to the document, 0 should freeze the backgroundsubtractor, but it fails. So we put very small number
image = cv.copyMakeBorder(image, istream.half_ROI_size, istream.half_ROI_size, istream.half_ROI_size, istream.half_ROI_size, cv.BORDER_CONSTANT, value=[0, 0, 0])


dilate_size=2
erosion_size=4

#masked_image = cv.floodFill(masked_image, seedPoint=(0,0))

# 1. Dilate
dilate_shape = cv.MORPH_RECT
element_dlt = cv.getStructuringElement(dilate_shape, (2*dilate_size+1,2*dilate_size+1),(dilate_size, dilate_size))
masked_image_dlt = cv.dilate(masked_image, element_dlt)


# 2. Erode :
erosion_shape = cv.MORPH_RECT
element_erd = cv.getStructuringElement(erosion_shape, (2*erosion_size+1,2*erosion_size+1),(erosion_size, erosion_size))
masked_image_dlt_erd = cv.erode(masked_image_dlt,element_erd)

parameter = cv.SimpleBlobDetector_Params()
parameter.filterByArea = True
parameter.filterByConvexity = True
parameter.filterByInertia = False
parameter.filterByColor = False
parameter.minArea = 500  # this value defines the minimum size of the blob
parameter.maxArea = 10000  # this value defines the maximum size of the blob
parameter.minDistBetweenBlobs = 1
parameter.minConvexity = 0.2


detector = cv.SimpleBlobDetector_create(parameter)
detected_blob = detector.detect(masked_image_dlt_erd)
print(len(detected_blob))


fig = plt.figure(1)
fig.clf()
ax = fig.subplots(2, 2)
ax[0,0].imshow(image)
ax[0,0].set_title('Original')
ax[0,1].imshow(masked_image)
ax[0,1].set_title('Masked_image')
ax[1,0].imshow(masked_image_dlt)
ax[1,0].set_title('Dilate')
ax[1,1].imshow(masked_image_dlt_erd)
ax[1,1].set_title('Dilate + Erode')


max_blob_index = 0
if len(detected_blob) > 1:# if multiple blob is detected, select the largest one
    max_blob_size = 0
    for i, blob in enumerate(detected_blob):
        if max_blob_size < blob.size:
            max_blob_size = blob.size
            max_blob_index = i
    print('Multiple blobs : %d detected' % len(detected_blob))
    showChosenImage = True
blob_center_row, blob_center_col = int(np.round(detected_blob[max_blob_index].pt[1])) , int(np.round(detected_blob[max_blob_index].pt[0]))

chosen_image = image[
               blob_center_row - istream.half_ROI_size + istream.half_ROI_size : blob_center_row + istream.half_ROI_size + istream.half_ROI_size,
               blob_center_col - istream.half_ROI_size + istream.half_ROI_size : blob_center_col + istream.half_ROI_size + istream.half_ROI_size,:]

plt.figure(2)
plt.clf()
plt.imshow(chosen_image)

