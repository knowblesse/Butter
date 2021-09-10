import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from threading import Thread
from queue import Queue
import time

class ROI_image_stream():
    def __init__(self,path_data, ROI_size, threshold=80):
        """
        __init__ : initialize ROI image extraction stream object
        """
        # Parameter
        self.path_data = path_data
        if self.path_data.suffix == '.mkv': # path is video.mkv
            self.path_video = self.path_data
        elif self.path_data.suffix == '.avi': # path is video.avi
            self.path_video = self.path_data
        elif sorted(self.path_data.glob('*.mkv')): # path contains video.mkv
            self.path_video = next(self.path_data.glob('*.mkv'))
            print('ROI_image_stream : found *.mkv')
        elif sorted(self.path_data.glob('*.avi')): # path contains video.avi
            self.path_video = next(self.path_data.glob('*.avi'))
            print('ROI_image_stream : found *.avi')
        else:
            print('ROI_image_stream : Can not find video file in %s', self.path_data)
        # stream status 
        self.isBackgroundSubtractorTrained = False
        self.threshold = threshold
        self.isMultithreading = False

        # setup VideoCapture
        self.vid = cv.VideoCapture(str(self.path_video))
        self.num_frame = self.vid.get(cv.CAP_PROP_FRAME_COUNT)
        
        # ROI size
        self.ROI_size = ROI_size
        self.half_ROI_size = int(np.round(self.ROI_size/2))
        if ROI_size%2 != 0 :
            raise(BaseException('ROI_size is not dividable with 2!'))
        # TODO Delete
        self.multipleBlobs = 0
        self.noBlob = 0

    def trainBackgroundSubtractor(self, stride=2000, start_empty=False):
        """
        trainBackgroundSubtractor : train BackgroundSubtractorKNN
        ---------------------------------------------------------------- 
        stride : int : frame stride for training. large number is faster, but inaccurate.
        start_empty : bool : is animal present from the beginning of the video. 
            True uses high learning rate for initial 1 sec video.
        """
        self.backSub = cv.createBackgroundSubtractorKNN()
        self.backSub.setShadowThreshold(0.01)
        self.backSub.setShadowValue(0)
        if start_empty:
            for i in np.arange(self.vid.get(cv.CAP_PROP_FPS)):
                self.vid.set(cv.CAP_PROP_POS_FRAMES, i)
                ret, image = self.vid.read()
                self.backSub.apply(cv.threshold(image, self.threshold, 0, cv.THRESH_TOZERO)[1], learningRate=0.2)
        print('ROI_image_stream : Start Background training')
        for frame in tqdm(np.arange(0, self.num_frame, stride)):
            self.vid.set(cv.CAP_PROP_POS_FRAMES, frame)
            ret, image = self.vid.read()
            if image is None:
                break
            else:
                self.backSub.apply(cv.threshold(image, self.threshold, 0, cv.THRESH_TOZERO)[1], learningRate=0.05)
        print('ROI_image_stream : Background training Complete')
        self.isBackgroundSubtractorTrained = True

    def getFrame(self, frame_number):
        """
        getFrame : return original frame
        ----------------------------------------------------------------
        frame_number : int : frame to process
        ----------------------------------------------------------------
        returns frame 
        """
        self.vid.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, image = self.vid.read()
        if not ret:
            raise(BaseException(f'ROI_image_stream : Can not retrieve frame # {frame_number}'))
        return image

    def drawFrame(self, frame_number):
        """
        drawFrame : draw original frame
        --------------------------------------------------------------------------------
        frame_number : int : frame to process
        """
        fig = plt.figure(1)
        fig.clf()
        ax = fig.subplots(1,1)
        ax.imshow(self.getFrame(frame_number))
        ax.set_title('Frame : %05d' % frame_number,fontdict={'fontsize': 15})

    def startROIextractionThread(self, frame_number_array):
        """
        startROIextractionThread : start ROI extraction Thread for continuous processing.
            When called, two thread (video read and opencv ROI detection) is initiated.
            Processed ROI is stored in self.roiDetectionQ 
        --------------------------------------------------------------------------------
        frame_number_array : 1D array : frame numbers to process
        """
        # Video IO Thread and Queue
        self.frameQ = Queue(maxsize=200)
        self.frame_number_array = frame_number_array
        self.vidIOthread = Thread(target=self.__readVideo, args=())
        self.vidIOthread.daemon = True # indicate helper thread
        self.frame_number_array_idx = 0
        if not self.vidIOthread.isAlive():
            self.vidIOthread.start()

        # ROI detection Thread and Queue
        self.blobQ = Queue(maxsize=200)
        self.roiDetectionThread = Thread(target=self.__processFrames, args=())
        self.roiDetectionThread.daemon = True
        self.roiDetectionThread.start()

        self.isMultithreading = True

    def __readVideo(self):
        """
        __readVideo : multithreading. read video, extract frame listed in self.frame_number_array and store in self.frameQ
        """
        print('ROI_image_stream : Video IO Thread started')
        while True:
            if not self.frameQ.full():
                frame_number = self.frame_number_array[self.frame_number_array_idx]
                self.vid.set(cv.CAP_PROP_POS_FRAMES, frame_number)
                ret, image = self.vid.read()
                self.frameQ.put((frame_number, ret, image))
                self.frame_number_array_idx += 1
            else:
                time.sleep(0.5)
            if self.frame_number_array_idx >= self.frame_number_array.shape[0]:
                break
        print('ROI_image_stream : Video IO Thread stopped')

    def __processFrames(self):
        """
        __processFrames : multithreading. extract ROI from frame stored in self.frameQ and store in self.blobQ
        """
        print('ROI_image_stram : ROI extraction Thread started')
        # run until frameQ is empty and thread is dead 
        while not(self.frameQ.empty()) or self.vidIOthread.isAlive():
            if not self.blobQ.full():
                frame_number, ret, image = self.frameQ.get()
                detected_blob = self.__findBlob(image)
                self.blobQ.put((frame_number, image, detected_blob))
            else:
                time.sleep(0.5)

    def __findBlob(self, image):
        """
        __findBlob: from given image, applay noise filter and find blob.
        --------------------------------------------------------------------------------
        image : 3D np.array : image to process
        --------------------------------------------------------------------------------
        return blob array
        --------------------------------------------------------------------------------
        """
        # Process Image
        image = cv.threshold(image, self.threshold, 0, cv.THRESH_TOZERO)[1]
        # according to the doc, 0 should freeze the backsub, but it fails So we put small value
        masked_image = self.backSub.apply(image, learningRate=1e-6)

        def getKernel(size):
            size = int(size)
            return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size),
                                            ((int((size - 1) / 2), int((size - 1) / 2))))

        # opening -> delete noise : erode and dilate
        # closing -> make into big object : dilate and erode
        denoised_mask = masked_image
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, getKernel(3))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, getKernel(5))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_CLOSE, getKernel(10))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, getKernel(7))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, getKernel(12))

        # Detect Blob
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
        parameter.minThreshold = 253
        parameter.maxThreshold = 255
        parameter.thresholdStep = 1
        detector = cv.SimpleBlobDetector_create(parameter)

        # soft detector
        parameter.minConvexity = 0.15
        parameter.minCircularity = 0.15

        detector_soft = cv.SimpleBlobDetector_create(parameter)

        # detect
        detected_blob = detector.detect(denoised_mask)

        if len(detected_blob) == 0: # if not found, try softer one
            detected_blob = detector_soft.detect(denoised_mask)

        return detected_blob

    def getROIImage(self, frame_number=-1, previous_rc = []):
        """
        extractROIImage : return ROI frame from the video
            frame_number can be omitted if the class uses multithreading.
            In that case, self.startROIextractionThread function must be called prior to this func.
            If frame_number is not provided and the multithreading is not enabled, returns error.
       -------------------------------------------------------------------------------- 
        frame_number : int or int numpy array : frame to process
        ---------------------------------------------------------------- 
        outputFrame : 3D numpy array : ROI frame  
        center : (r,c) : location of the center of the blob
        """
        if not self.isBackgroundSubtractorTrained:
            raise(BaseException('BackgroundSubtractor is not trained'))

        if frame_number != -1: # frame_number is provided
            self.vid.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            ret, image = self.vid.read() # the part of "image" will be returned and "masked_image" is used to center of ROI detection
            detected_blob = self.__findBlob(image)

        else: # frame number is not provided
            if not self.isMultithreading: # if multithreading is not used
                raise(TypeError('getROIImage() missing 1 required positional argument: \'frame_number\'\n'
                                'If you are trying to use this function as multithreading, check if you called startROIextractionThread()'))
            frame_number, image, detected_blob = self.blobQ.get()

        # Further process detected blobs
        if len(detected_blob) == 0 :
            self.noBlob += 1
            raise(BlobDetectionFailureError(f'ROI_image_stream : No blob is detected from frame {frame_number}. {self.noBlob}'))
        elif len(detected_blob) > 1:# if multiple blob is detected, select the largest one
            self.multipleBlobs += 1
            final_blob_index = 0
            if previous_rc != []: # use previous point to select blob
                min_blob_distance = 1000000000
                for i, blob in enumerate(detected_blob):
                    distance = ((previous_rc[0] - blob.pt[1])**2 +  (previous_rc[1] - blob.pt[0])**2)
                    if min_blob_distance > distance:
                        min_blob_distance = distance
                        final_blob_index = i
            else: # use size to select blob
                max_blob_size = 0
                for i, blob in enumerate(detected_blob):
                    if max_blob_size < blob.size:
                        max_blob_size = blob.size
                        final_blob_index = i
            detected_blob = [detected_blob[final_blob_index]]
            #print(f'ROI_image_stream : Multiple blobs({len(detected_blob)}) detected in frame {frame_number}. The largest one is selected. {self.multipleBlobs}')

        blob_center_row, blob_center_col = int(np.round(detected_blob[0].pt[1])) , int(np.round(detected_blob[0].pt[0]))

        # Create expanded version of the original image.
        # In this way, we can prevent errors when center of the ROI is near the boarder of the image.
        expanded_image = cv.copyMakeBorder(image, self.half_ROI_size, self.half_ROI_size, self.half_ROI_size,
                                           self.half_ROI_size,
                                           cv.BORDER_CONSTANT, value=[0, 0, 0])

        chosen_image = expanded_image[
                       blob_center_row - self.half_ROI_size + self.half_ROI_size : blob_center_row + self.half_ROI_size + self.half_ROI_size,
                       blob_center_col - self.half_ROI_size + self.half_ROI_size : blob_center_col + self.half_ROI_size + self.half_ROI_size,:]

        return (chosen_image, [blob_center_row, blob_center_col])

class BlobDetectionFailureError(Exception):
    """Error class for blob detection"""

def vector2degree(r1,c1,r2,c2):
    l = ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5
    if r1 < r2: # 0 ~180
        return np.rad2deg(np.arccos((c2-c1) / l))
    elif r1 > r2: # 180 ~ 360
        return 360 - np.rad2deg(np.arccos((c2 - c1) / l))
    else:
        if c2 > c1:
            return 0
        else:
            return 180
