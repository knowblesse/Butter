"""
Detection.py
2021 Knowblesse
Run Detection with new file
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from helper_function import ROI_image_stream
from helper_function import checkDataSet
import time
from tqdm import tqdm

################################################################
# Constants
################################################################
path_dataset = Path.home() / 'VCF/butter/dataset/Ex1/0423_04_Valium_0.2/'
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

new_model = keras.models.load_model('210617_working_model')

################################################################j
# Check the starting and ending part of the video
################################################################
num_frame = int(istream.vid.get(cv.CAP_PROP_FRAME_COUNT))
fps = istream.vid.get(cv.CAP_PROP_FPS)
time_sec = num_frame / fps
print("Total Frame : %05d | Running Time : %02d m %02d s" % (num_frame, np.floor(time_sec/60), np.remainder(time_sec,60))) 
istream.drawFrame(140)

START_FRAME = 140
END_FRAME = int(np.min((np.round(START_FRAME + 600*fps), num_frame-2)))
istream.drawFrame(END_FRAME)

################################################################
# Run
################################################################
FPS = 3
output_data = np.zeros((np.arange(START_FRAME, END_FRAME, int(fps/3)).shape[0],4))
start_time = time.time()
for i, idx in enumerate(tqdm(np.arange(START_FRAME, END_FRAME,int(fps/3)))):
    chosen_image, coor = istream.extractROIImage(idx)
    testing = np.expand_dims(chosen_image,0) / 255
    result = new_model.predict(testing)
    output_data[i, :] = [coor[0] + result[0,0], coor[1] + result[0,1], result[0, 2] - result[0, 0], result[0, 3] - result[0, 1]]
end_time = time.time()

################################################################
# Save Output
################################################################
# Total numbe of frame in the video, video fps, detector fps, start frame number
np.savetxt(istream.path_video.stem + '_output.csv',np.vstack(([num_frame, fps, FPS, START_FRAME],output_data)),'%d',delimiter='\t')
