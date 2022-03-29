import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ROI_image_stream import ROI_image_stream
from tqdm import tqdm

# Load Test Videos
video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21AUG3/21AUG3-211026-171617')

istream = ROI_image_stream(video_path, 224)

istream.buildForegroundModel()
istream.startROIextractionThread(np.arange(0, istream.vc.get(cv.CAP_PROP_FRAME_COUNT), 5))

while True:
    chosen_image, [blob_center_row, blob_center_col] = istream.getROIImage()
    cv.imshow('ROI', chosen_image)
    cv.waitKey(5)