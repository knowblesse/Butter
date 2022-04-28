
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from ROI_image_stream import *

################################################################
# Constants
################################################################
video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210730-181651_IL')
base_network = 'mobilenet'

################################################################
# Setup
################################################################
istream = ROI_image_stream(video_path,ROI_size=224)

istream.buildForegroundModel()

istream.drawROI(6955)