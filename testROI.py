import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ROI_image_stream import ROI_image_stream
from tqdm import tqdm

# Load Test Videos
video_path_light_change = Path('TestVideo_Normal.avi')

istream_light_change = ROI_image_stream(video_path_light_change, 224)

istream_light_change.trainBackgroundSubtractor()

frame_number = 20


for frame_number in np.arange(0, 3000, 100):
    img = istream_light_change.getFrame(frame_number)
    ROIImage, ROIcenter = istream_light_change.getROIImage(frame_number)
    cv.rectangle(img,
                 [ROIcenter[0] - istream_light_change.half_ROI_size, ROIcenter[1] - istream_light_change.half_ROI_size],
                 [ROIcenter[0] + istream_light_change.half_ROI_size, ROIcenter[1] + istream_light_change.half_ROI_size],
                 (255, 0, 0), thickness=2)
    print(ROIcenter)
    plt.figure(1)
    plt.clf()
    plt.imshow(img)
    plt.draw()
    plt.figure(2)
    plt.clf()
    plt.imshow(ROIImage)
    plt.draw()
    plt.pause(0.05)