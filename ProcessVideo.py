"""
ProcessVideo.py
@Knowblesse 2021
21 AUG 31
Process new video and return row,col coordinates with head direction
- Inputs
    -video_path : path to the to-be-labeled video
- Outputs
    - row, col, degree 2D np matrix
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import time
from tqdm import tqdm

def ProcessVideo(video_path, model_path):
    if not(type(video_path) is pathlib.PosixPath):
        raise(BaseException('ProcessVideo : video_path should be pathlib.Path object'))
    if not(type(model_path) is pathlib.PosixPath):
        raise(BaseException('ProcessVideo : model_path should be pathlib.Path object'))

    # Load Model
    try:
        Model = keras.models.load_model(str(model_path))
    except:
        raise(BaseException('ProcessVideo : Can not load model from ' + str(model_path)))

    # Get ROI size from the loaded model
    ROI_size = Model.layers[0].input.shape[1]

    # Train ROI_image_stream
    istream = ROI_image_stream(video_path, ROI_size=ROI_size)
    istream.trainBackgroundSubtractor(stride=500)

    # Print Video Info
    num_frame = int(istream.vid.get(cv.CAP_PROP_FRAME_COUNT))
    fps = istream.vid.get(cv.CAP_PROP_FPS)
    time_sec = num_frame / fps
    print(f"ProcessVideo : Video Info : {num_frame:05d}frames : {np.floor(time_sec/60):d} m {np.remainder(time_sec,60):d} s")

    # TODO : Find out when to start
    # TODO : Find out when to stop

    istream.drawFrame(140)

    START_FRAME = 140
    END_FRAME = int(np.min((np.round(START_FRAME + 600*fps), num_frame-2)))
    istream.drawFrame(END_FRAME)

    # TODO : Print estimated processing time

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
    # TODO : if ROI is not found, compensate or ask user to manually select ROI
    # TODO : detect anormaly

    ################################################################
    # Save Output
    ################################################################
    # Total numbe of frame in the video, video fps, detector fps, start frame number
    np.savetxt(istream.path_video.stem + '_output.csv',np.vstack(([num_frame, fps, FPS, START_FRAME],output_data)),'%d',delimiter='\t')
