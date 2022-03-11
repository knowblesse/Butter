import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ROI_image_stream import ROI_image_stream
from tqdm import tqdm

# Load Test Videos
video_path_normal = Path('TestVideo_Normal.avi')
video_path_blinking = Path('TestVideo_Blinking.avi')
video_path_light_change = Path('TestVideo_LightChange.avi')

istream_normal = ROI_image_stream(video_path_normal, 224)
istream_blinking = ROI_image_stream(video_path_blinking, 224)
istream_light_change = ROI_image_stream(video_path_light_change, 224)

num_frames2use = 200

frameStorage_normal = np.zeros((istream_normal.frame_size[0], istream_normal.frame_size[1], 3, num_frames2use), dtype=np.uint8)
for i, frame in enumerate(tqdm(np.round(np.linspace(0, istream_normal.num_frame-1, num_frames2use)).astype(int))):
    image = istream_normal.getFrame(frame)
    image = cv.bitwise_and(image, image, mask=istream_normal.global_mask)
    frameStorage_normal[:,:,:,i] = image

frameStorage_blinking = np.zeros((istream_blinking.frame_size[0], istream_blinking.frame_size[1], 3, num_frames2use), dtype=np.uint8)
for i, frame in enumerate(tqdm(np.round(np.linspace(0, istream_blinking.num_frame-1, num_frames2use)).astype(int))):
    image = istream_blinking.getFrame(frame)
    image = cv.bitwise_and(image, image, mask=istream_blinking.global_mask)
    frameStorage_blinking[:,:,:,i] = image

frameStorage_lightchange = np.zeros((istream_light_change.frame_size[0], istream_light_change.frame_size[1], 3, num_frames2use), dtype=np.uint8)
for i, frame in enumerate(tqdm(np.round(np.linspace(0, istream_light_change.num_frame-1, num_frames2use)).astype(int))):
    image = istream_light_change.getFrame(frame)
    image = cv.bitwise_and(image, image, mask=istream_light_change.global_mask)
    frameStorage_lightchange[:,:,:,i] = image


## Forground model building
animalSizes = np.zeros(num_frames2use)
animalThreshold = np.zeros(num_frames2use)
animalConvexity = np.zeros(num_frames2use)
animalCircularity = np.zeros(num_frames2use)

medianImage = np.median(frameStorage_lightchange, axis=3).astype(np.uint8)
for i in tqdm(np.arange(num_frames2use)):
    image = cv.cvtColor(cv.absdiff(frameStorage_lightchange[:,:,:,i], medianImage), cv.COLOR_RGB2GRAY)
    animalThreshold[i] = np.quantile(image, 0.99)
    binaray_image = cv.threshold(image,animalThreshold[i], 255, cv.THRESH_BINARY)[1]

    # Find the largest contour
    cnts = cv.findContours(binaray_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
    maxCntSize = 0
    maxCntIndex = None
    for j, cnt in enumerate(cnts):
        area = cv.contourArea(cnt)
        if area > maxCntSize:
            maxCntSize = area
            maxCntIndex = j

    # Calculate feature information from the largest contour
    animalSizes[i] = cv.contourArea(cnts[maxCntIndex])
    convexHull = cv.convexHull(cnts[maxCntIndex])
    print(convexHull)

plt.figure(4)
plt.clf()
i = 199
image = cv.cvtColor(cv.absdiff(frameStorage_lightchange[:,:,:,i], medianImage), cv.COLOR_RGB2GRAY)
animalThreshold[i] = np.quantile(image, 0.99)
binaray_image = cv.threshold(image,animalThreshold[i], 255, cv.THRESH_BINARY)[1]
# Find the largest contour
cnts = cv.findContours(binaray_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
maxCntSize = 0
maxCntIndex = None
for j, cnt in enumerate(cnts):
    area = cv.contourArea(cnt)
    if area > maxCntSize:
        maxCntSize = area
        maxCntIndex = j
plt.imshow(image)

cv.drawContours(image, [cnts[maxCntIndex]], 0, (100,0,0), 2)
cv.drawContours(image, [cv.convexHull(cnts[maxCntIndex])], 0, (255,0,0), 2)
plt.imshow(image)




# Get Diff
"""
frameStorage_blinking : i = 77, 145~ frames has some defects.
"""
medianImage = np.median(frameStorage_lightchange, axis=3).astype(np.uint8)
i = 30
forImage = cv.cvtColor(cv.absdiff(frameStorage_lightchange[:,:,:,i], medianImage), cv.COLOR_RGB2GRAY)
# forImage = cv.cvtColor(
#     cv.absdiff(frameStorage_lightchange[:, :, :, i], np.median(frameStorage_lightchange, axis=3).astype(np.uint8)),
#     cv.COLOR_RGB2GRAY)
plt.figure(1)
plt.clf()
plt.imshow(forImage)
plt.title(str(i))

# Take only the uppper 1% of the diff value
forImage2 = cv.threshold(forImage,thr, 255, cv.THRESH_BINARY)[1]
plt.figure(2)
plt.clf()
plt.imshow(forImage2)
plt.title('threshold image')

# Find contour
result = cv.findContours(forImage2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
forImage3 = np.zeros(forImage2.shape)
maxCntSize = 0
maxCnt = 0
for cnt in result[0]:
    area = cv.contourArea(cnt)
    if  area > maxCntSize:
        maxCntSize = area
        maxCnt = cnt
    cv.drawContours(forImage3, [cnt], 0, (100, 0, 0), 3)
cv.drawContours(forImage3, [maxCnt], 0, (255, 0, 0), 3)
plt.figure(3)
plt.clf()
plt.imshow(forImage3)
plt.title('contours : ' + str(len(result[0])))
plt.draw()
plt.pause(0.1)






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
parameter.minThreshold = 100
parameter.maxThreshold = 255
parameter.thresholdStep = 1
detector = cv.SimpleBlobDetector_create(parameter)
a = detector.detect()



