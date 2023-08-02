"""
Butter.py
@Knowblesse 2021
21 AUG 31
Process new video and return row,col coordinates with head direction
- Inputs
    -video_path : path to the to-be-labeled video
- Outputs
    - row, col, degree 2D np matrix
"""
import os
from collections import namedtuple
from collections import deque
import queue

import tensorflow as tf
from tensorflow import keras

from ROI_image_stream import *
from butterUtil import vector2degree

class Butter:
    """
    Process New Video. Find ROI and detect head location and degree in the ROI.
    """
    def __init__(self, video_path, model_path, process_fps=5, predictBatchSize=100, roiCoordinateData=[]):
        """
        video_path : pathlib.PosixPath(Path) : target video path
        model_path : pathlib.PosixPath(Path) : pre-trained model path
        process_fps : int : number of frame to process per second. 
        """
        print(f'Butter : {video_path.stem}')
        if not(issubclass(type(video_path), Path)):
            raise(BaseException('Butter : video_path should be pathlib.Path object'))
        if not(issubclass(type(video_path), Path)):
            raise(BaseException('Butter : model_path should be pathlib.Path object'))
        self.video_path = video_path
        self.model_path = model_path

        self.isStartPositionChecked = False
        self.isProcessed = False

        # Batch Size
        self.predictBatchSize = predictBatchSize

        # Load Model
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable TF message
        print('Butter : Loading Model...')
        try:
            model = keras.models.load_model(str(model_path), )
        except:
            raise(BaseException('Butter : Can not load model from ' + str(model_path)))
        self.model = model
        print('Butter : Model Loaded')

        # Check if the model has the head detection function
        if self.model.output.shape[1] == 2:
            self.isHeadDetectionEnabled = False
            print('Butter : Model does not have head detection function')
        elif self.model.output.shape[1] == 4:
            self.isHeadDetectionEnabled = True
            print('Butter : Model have head detection function')

        # Get ROI size from the loaded model
        self.ROI_size = model.layers[0].input.shape[1]

        # Train ROI_image_stream
        if len(roiCoordinateData) > 0:
            self.istream = ROI_image_stream_noblob(video_path, ROI_size=self.ROI_size)
            self.istream.setRoiCoordinateData(roiCoordinateData)
            self.start_frame = roiCoordinateData[0,0]
            self.isStartPositionChecked = True
        else:
            self.istream = ROI_image_stream(video_path, ROI_size=self.ROI_size)

        # Print Video Info
        self.process_fps = process_fps
        self.num_frame = self.istream.num_frame
        time_sec = self.num_frame / self.istream.fps
        print(f"Butter : Video Info : {self.num_frame:05d} frames : {int(np.floor(time_sec/60)):d} m {int(np.remainder(time_sec,60)):d} s")

    def setGlobalMask(self, mask=[]):
        # Check type
        if mask:
            if (type(mask) != np.ndarray) or (mask.dtype != np.uint8):
                raise TypeError('Butter : Wrong type of global mask. Only accept numpy.ndarray with uint8 elements')
        self.istream.setGlobalMask(mask)

    def getGlobalMask(self):
        return self.istream.global_mask

    def saveSampleFrames(self, num_frames2use=100):
        self.istream.saveSampleFrames(num_frames2use=num_frames2use)

    def setForegroundModel(self, foregroundModel):
        self.istream.setForegroundModel(foregroundModel)

    def getForegroundModel(self):
        return self.istream.foregroundModel
    def buildForegroundModel(self):
        self.istream.buildForegroundModel(verbose=True)
        self.num_frame = self.istream.num_frame

    def buildBackgroundModel(self):
        self.istream.buildBackgroundModel()
    def setBackgroundModel(self, background):
        self.istream.setBackgroundModel(background)
    def getBackgroundModel(self, draw=True):
        if draw:
            cv.imshow('Background', self.istream.background)
            cv.waitKey()
            cv.destroyWindow('Background')
        return self.istream.background
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
            print(f'Butter : Frame {start_frame:d}')
            cv.imshow('First Image Detection', chosen_image)
            cv.waitKey(1)
            prompt = int(input('Butter : Found? (0: Found, not zero: skip frame)'))
            if prompt == 0 :
                cv.destroyWindow('First Image Detection')
                break
            else:
                start_frame += int(self.process_fps * prompt)
        self.start_frame = start_frame
        self.isStartPositionChecked = True
        print(f'Butter : Starting from Frame {self.start_frame:d}')

    def run(self):
        roi_batch = ROI_Batch(self.predictBatchSize)
        if not self.isStartPositionChecked:
            raise(BaseException('Butter : check Start Position First!'))
        self.output_data = np.zeros((np.arange(self.start_frame, self.num_frame, self.process_fps).shape[0],4), dtype=int)
        cumerror = 0

        # set for multiprocessing. reading frame automatically starts from this function
        self.istream.startROIextractionThread(self.start_frame, stride=self.process_fps)

        # For every process frames
        for idx, frameNumber in enumerate(tqdm(np.arange(self.start_frame, self.num_frame, self.process_fps))):
            try:
                image, coor = self.istream.getROIImage()
                roi_batch.push(idx, frameNumber, image, coor)
            except queue.Empty:
                print('Butter : Video ended prematurely.')
                self.output_data = self.output_data[0:int(self.istream.num_frame/self.process_fps)+1,:] # shrink size
                self.num_frame = self.istream.num_frame
                break
            except BlobDetectionFailureError:
                cumerror += 1
                print(f'Butter : Couldn\'t find the ROI in Frame {frameNumber}. Total {cumerror}')
                self.output_data[idx,:] = [frameNumber, -1, -1, -1]

            # Predict the batch if full.
            if roi_batch.full():
                batch = ROI(*zip(*roi_batch.popall()))
                testing = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(batch.image))
                result = self.model.predict(testing, verbose=0)
                if self.isHeadDetectionEnabled:
                    self.output_data[batch.idx, :] = np.concatenate((np.expand_dims(batch.frameNumber, 1), (
                                np.array(batch.coor) + result[:, :2] - int(self.ROI_size / 2)).astype(int),
                                                                     np.expand_dims(
                                                                         vector2degree(result[:, 0], result[:, 1],
                                                                                       result[:, 2], result[:, 3]), 1)),
                                                                    axis=1)
                else:
                    self.output_data[batch.idx, :] = np.concatenate((np.expand_dims(batch.frameNumber, 1), (
                                np.array(batch.coor) + result[:, :2] - int(self.ROI_size / 2)).astype(int),
                                                                     np.expand_dims(np.ones(result.shape[0]), 1)),
                                                                    axis=1)
        # ---- All frames are either processed or put into the roi_batch.
        # If unprocessed frame exist in the roi_batch, run for the last batch
        if len(roi_batch):
            batch = ROI(*zip(*roi_batch.popall()))
            testing = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(batch.image))
            result = self.model.predict(testing)
            if self.isHeadDetectionEnabled:
                self.output_data[batch.idx, :] = np.concatenate((np.expand_dims(batch.frameNumber, 1), (
                        np.array(batch.coor) + result[:, :2] - int(self.ROI_size / 2)).astype(int),
                                                                 np.expand_dims(
                                                                     vector2degree(result[:, 0], result[:, 1],
                                                                                   result[:, 2], result[:, 3]), 1)),
                                                                axis=1)
            else:
                self.output_data[batch.idx, :] = np.concatenate((np.expand_dims(batch.frameNumber, 1), (
                        np.array(batch.coor) + result[:, :2] - int(self.ROI_size / 2)).astype(int),
                                                                 np.expand_dims(np.ones(result.shape[0]), 1)),
                                                                axis=1)
        self.isProcessed = True

    def save(self, save_path=''):
        """
        save processed data
        --------------------------------------------------------------------------------
        save_path : pathlib.PosixPath object : save path. if nothing is provided, save next to target video
        """
        if not self.isProcessed:
            raise(BaseException('Butter : run the processor first!'))
        if save_path == '':
            save_path = self.video_path

        if save_path.is_dir():
            if sorted(save_path.glob('*.mkv')):
                video_path = next(save_path.glob('*.mkv'))
            elif sorted(save_path.glob('*.avi')):
                video_path = next(save_path.glob('*.avi'))
            elif sorted(save_path.glob('*.mp4')):
                video_path = next(save_path.glob('*.mp4'))
            elif sorted(save_path.glob('*.mpg')):
                video_path = next(save_path.glob('*.mpg'))
            else:
                raise(BaseException('Butter : Save path must be a file not directory'))
            txt_save_path = video_path.parent / (video_path.stem + '_buttered.csv')
        elif save_path.is_file():
            if save_path.suffix == '.csv':
                txt_save_path = save_path
            elif save_path.suffix == '.avi' or save_path.suffix == '.mkv':
                txt_save_path = save_path.parent / (save_path.stem + '_buttered.csv')
            else:
                raise(BaseException('Butter : Save path must end with .csv'))
        else:
            raise(BaseException('Butter : Unknown path'))

        np.savetxt(str(txt_save_path.absolute()),self.output_data,'%d',delimiter='\t')

    def checkResult(self, num_frame_to_check = 10):
        if not self.isProcessed:
            raise(BaseException('Butter : run the processor first!'))

        if type(num_frame_to_check) is np.ndarray:
            frames = num_frame_to_check
        else:
            frames = np.random.permutation(self.output_data[:,0])[0:num_frame_to_check]
        for frame in frames:
            self.istream.vc.set(cv.CAP_PROP_POS_FRAMES,frame)
            ret, img = self.istream.vc.read()
            if not ret:
                raise(BaseException(f'Butter : Can not retrieve frame # {frame}'))
            idx = np.where(self.output_data[:,0] == frame)[0]
            if self.output_data[idx,1] == -1:
                print(f'Butter : Frame {frame} is a ROI extraction failed frame. Skipping')
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


