import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ROI_image_stream import ROI_image_stream
from tqdm import tqdm

# Load Test Videos
video_path_light_change = Path('TestVideo_LightChange.avi')

istream_light_change = ROI_image_stream(video_path_light_change, 224)

istream_light_change.trainBackgroundSubtractor()

istream_light_change.getROIImage(30)