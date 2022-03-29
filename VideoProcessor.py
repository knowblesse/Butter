"""
VideoProcessor.py
@Knowblesse 2021
21 AUG 31
Process new video and return row,col coordinates with head direction
- Inputs
    -video_path : path to the to-be-labeled video
- Outputs
    - row, col, degree 2D np matrix
"""
from ROI_image_stream import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from tqdm import tqdm
from queue import Queue
from collections import namedtuple

class VideoProcessor:
    """
    Process New Video. Find ROI and detect head location and degree in the ROI.
    """
    def __init__(self, video_path, model_path, process_fps=5):
        """
        video_path : pathlib.PosixPath(Path) : target video path
        model_path : pathlib.PosixPath(Path) : pre-trained model path
        process_fps : int : number of frame to process per second. 
        """

        if not(type(video_path) is pathlib.PosixPath):
            raise(BaseException('VideoProcessor : video_path should be pathlib.Path object'))
        if not(type(model_path) is pathlib.PosixPath):
            raise(BaseException('VideoProcessor : model_path should be pathlib.Path object'))
        self.video_path = video_path
        self.model_path = model_path

        self.isStartPositionChecked = False
        self.isProcessed = False

        # Batch Size
        self.predictBatchSize = 100

        # Load Model
        try:
            model = keras.models.load_model(str(model_path))
        except:
            raise(BaseException('VideoProcessor : Can not load model from ' + str(model_path)))
        self.model = model

        # Get ROI size from the loaded model
        self.ROI_size = model.layers[0].input.shape[1]

        # Train ROI_image_stream
        self.istream = ROI_image_stream(video_path, ROI_size=self.ROI_size)

        # Print Video Info
        self.process_fps = process_fps
        self.num_frame = self.istream.getFrameCount()
        self.fps = self.istream.getFPS()
        time_sec = self.num_frame / self.fps
        print(f"VideoProcessor : Video Info : {self.num_frame:05d}frames : {int(np.floor(time_sec/60)):d} m {int(np.remainder(time_sec,60)):d} s")

    def buildForegroundModel(self):
        self.istream.buildForegroundModel()

    def checkStartPosition(self):
        """
        Find out starting position.
        --------------------------------------------------------------------------------
        Go through the video until a blob is found. 
        Show the blob and ask whether the found blob is the animal
        """
        #Find out when to start
        start_frame = 0
        while True:
            while True: # run until a Blob is found
                try:
                    (chosen_image, [blob_center_row, blob_center_col]) = self.istream.getROIImage(start_frame)
                except BlobDetectionFailureError:
                    start_frame += self.process_fps
                else:
                    break
            print(f'VideoProcessor : Frame {start_frame:d}')
            plt.imshow(chosen_image)
            plt.title(str(start_frame))
            plt.pause(1)
            prompt = int(input('VideoProcessor : Found? (0: Found, not zero: skip frame)'))
            if prompt == 0 :
                break
            else:
                start_frame += int(self.process_fps * prompt)
        self.start_frame = start_frame
        self.isStartPositionChecked = True
        print(f'VideoProcessor : Starting from Frame {self.start_frame:d}')

    def run(self):
        roi_batch = ROI_Batch(self.predictBatchSize)
        if not self.isStartPositionChecked:
            raise(BaseException('VideoProcessor : check Start Position First!'))
        self.output_data = np.zeros((np.arange(self.start_frame, self.num_frame, self.process_fps).shape[0],4), dtype=np.int)
        cumerror = 0

        # set for multiprocessing. reading frame automatically starts from this function
        self.istream.startROIextractionThread(np.arange(self.start_frame, self.num_frame, self.process_fps))

        for idx, frameNumber in enumerate(tqdm(np.arange(self.start_frame, self.num_frame, self.process_fps))):
            try:
                image, coor = self.istream.getROIImage()
                roi_batch.push(idx, frameNumber, image, coor)
            except BlobDetectionFailureError:
                cumerror += 1
                print(f'VideoProcessor : Couldn\'t find the ROI in Frame {frameNumber}. Total {cumerror}')
                self.output_data[idx,:] = [frameNumber, -1, -1, -1]

            # Predict the batch
            if roi_batch.full():
                batch = ROI(*zip(*roi_batch.popall()))
                testing = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(batch.image))
                result = self.model.predict(testing)
                self.output_data[batch.idx,:] = np.concatenate((np.expand_dims(batch.frameNumber,1), (np.array(batch.coor) + result[:,:2] - int(self.ROI_size/2)).astype(np.int), np.expand_dims(vector2degree(result[:,0], result[:,1], result[:,2], result[:,3]),1)), axis=1)

        # Run for the last batch
        print(f'Batch size : {len(roi_batch)}')
        batch = ROI(*zip(*roi_batch.popall()))
        testing = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(batch.image))
        result = self.model.predict(testing)
        self.output_data[batch.idx, :] = np.concatenate((np.expand_dims(batch.frameNumber, 1), (np.array(batch.coor) + result[:, :2] - int(self.ROI_size / 2)).astype(np.int), np.expand_dims(vector2degree(result[:, 0], result[:, 1], result[:, 2], result[:, 3]), 1)), axis=1)

        self.isProcessed = True

    def save(self, save_path=''):
        """
        save processed data
        --------------------------------------------------------------------------------
        save_path : pathlib.PosixPath object : save path. if nothing is provided, save next to target video
        """
        if not self.isProcessed:
            raise(BaseException('VideoProcessor : run the processor first!'))
        if save_path == '':
            save_path = self.video_path

        if save_path.is_dir():
            if sorted(save_path.glob('*.mkv')):
                video_path = next(save_path.glob('*.mkv'))
            elif sorted(save_path.glob('*.avi')):
                video_path = next(save_path.glob('*.avi'))
            elif sorted(save_path.glob('*.mp4')):
                video_path = next(save_path.glob('*.mp4'))
            else:
                raise(BaseException('VideoProcessor : Save path must be a file not directory'))
            txt_save_path = video_path.parent / (video_path.stem + '_buttered.csv')
        elif save_path.is_file():
            if save_path.suffix == '.csv':
                txt_save_path = save_path
            elif save_path.suffix == '.avi' or save_path.suffix == '.mkv':
                txt_save_path = save_path.parent / (save_path.stem + '_buttered.csv')
            else:
                raise(BaseException('VideoProcessor : Save path must end with .csv'))
        else:
            raise(BaseException('VideoProcessor : Unknown path'))

        np.savetxt(str(txt_save_path.absolute()),self.output_data,'%d',delimiter='\t')

    def checkResult(self, num_frame_to_check = 10):
        if not self.isProcessed:
            raise(BaseException('VideoProcessor : run the processor first!'))

        if type(num_frame_to_check) is np.ndarray:
            frames = num_frame_to_check
        else:
            frames = np.random.permutation(self.output_data[:,0])[0:num_frame_to_check]
        for frame in frames:
            self.istream.vc.set(cv.CAP_PROP_POS_FRAMES,frame)
            ret, img = self.istream.vc.read()
            if not ret:
                raise(BaseException(f'VideoProcessor : Can not retrieve frame # {frame}'))
            idx = np.where(self.output_data[:,0] == frame)[0]
            if self.output_data[idx,1] == -1:
                print(f'VideoProcessor : Frame {frame} is a ROI extraction failed frame. Skipping')
                continue
            img = cv.circle(img, np.round(self.output_data[idx,2:0:-1]).astype(int)[0], 5, [0, 0, 255], -1)
            cv.line(img,
                    np.concatenate((np.round(self.output_data[idx, 2]), np.round(self.output_data[idx, 1]))).astype(int),
                    np.concatenate((
                        np.round(self.output_data[idx, 2] + 50 * np.cos(np.deg2rad(self.output_data[idx, 3]))),
                        np.round(self.output_data[idx, 1] + 50 * np.sin(np.deg2rad(self.output_data[idx, 3]))))).astype(
                        int),
                    [35, 139, 251], 3)
            cv.imshow('frame', img)
            cv.waitKey(2000)
        cv.destroyWindow('frame')

ROI = namedtuple('ROI', ('idx', 'frameNumber', 'image', 'coor'))
class ROI_Batch(object):
    def __init__(self, capacity):
        self.item = deque([], maxlen=capacity)
    def push(self, *args):
        self.item.append(ROI(*args))
    def __len__(self):
        return len(self.item)
    def full(self):
        if len(self.item) == self.item.maxlen:
            return True
        else:
            return False
    def clear(self):
        self.item.clear()
    def popall(self):
        items = list(self.item)
        self.clear()
        return items


