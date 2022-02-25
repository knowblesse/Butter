import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from threading import Thread
from queue import Queue
import time

class ROI_image_stream():
    def __init__(self,path_data, ROI_size, setMask=True, stableBackground=False):
        """
        __init__ : initialize ROI image extraction stream object
        ****************************************************************
        path_data : Path object : path of the video or a folder containing the video
        ROI_size : weight/height of the extracting ROI. This must be an even number.
        setMask : if True, ROI selection screen appears.
        stableBackground : if True, only the median image is used as the background.
        """
        # Setup VideoCapture
        self.path_video = self.__readVideoPath(path_data)
        self.vc = cv.VideoCapture(str(self.path_video))
        self.frame_size = (int(self.vc.get(cv.CAP_PROP_FRAME_HEIGHT)), int(self.vc.get(cv.CAP_PROP_FRAME_WIDTH)))
        self.num_frame = self.__getFrameSize(self.vc)

        # stream status 
        self.isBackgroundSubtractorTrained = False
        self.isMultithreading = False

        # ROI size
        self.ROI_size = ROI_size
        self.half_ROI_size = int(np.round(self.ROI_size/2))
        if ROI_size%2 != 0 :
            raise(BaseException('ROI_size is not dividable with 2!'))

        # Background Subtractor Parameters
        self.masked_image = []
        self.backSub_lr = 0
        self.stableBackground = stableBackground

        # BlobDetector
        # TODO : automatically find the best parameter
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
        self.detector = cv.SimpleBlobDetector_create(parameter)

        # soft detector
        parameter.minConvexity = 0.15
        parameter.minCircularity = 0.15

        self.detector_soft = cv.SimpleBlobDetector_create(parameter)

        # Set Mask
        if setMask:
            mask_position = cv.selectROI('Select ROI', self.getFrame(0))
            cv.destroyWindow('Select ROI')
            self.global_mask = np.zeros(self.frame_size, dtype=np.uint8)
            self.global_mask[mask_position[1]:mask_position[1]+mask_position[3], mask_position[0]:mask_position[0]+mask_position[2]] = 255
        else:
            self.global_mask = 255 * np.ones(self.frame_size, dtype=np.uint8)

    def trainBackgroundSubtractor(self, num_frames2use = 200):
        """
        trainBackgroundSubtractor : train BackgroundSubtractorKNN for initial movement detection
        ----------------------------------------------------------------------------------------
        training the background subtractor
        1. read (numModelFrame) number of frames across the video
        2. store these frames
        3. use np.quantile to generate imageMedian image
        4. subtract imageMedian from stored frames
        5. remainder is the animal containing frame
        6. calculate threshold from animal frame
        7. get mean animal size
        8. train background subtractor
        """
        self.backSub = cv.createBackgroundSubtractorMOG2()
        self.backSub.setNMixtures(3) #default : 5 : 30
        self.backSub.setHistory(20) # default : 500 : 100
        self.backSub.setVarThreshold(50) # default : 16 : 50
        self.backSub.setDetectShadows(False)

        # Store Multiple Frames for further processing
        storeFrame = np.zeros((self.frame_size[0], self.frame_size[1], 3, num_frame2use), dtype=np.uint8)

        stride = np.ceil(self.num_frame / num_frame2use)

        print('ROI_image_stream : Frame analysis...')
        for i, frame in enumerate(tqdm(np.arange(0, self.num_frame-1, stride))): 
            image = self.getFrame(frame)
            if image is None:
                break
            else:
                image = cv.bitwise_and(image, image, mask=self.global_mask)
                storeFrame[:,:,:,i] = image

        # Extract median image + background distribution

        print('ROI_image_stream : Foreground Model building...')
        #imageMedian = np.median(storeFrame,axis=3).astype(np.uint8)
        imageMedian = np.quantile(storeFrame, 0.2, axis=3).astype(np.uint8)
        self.imageMedian = imageMedian
        animalFrame = np.zeros((self.frame_size[0], self.frame_size[1], numModelFrame), dtype=np.uint8)
        animalSize = np.zeros(numModelFrame)
        for i in tqdm(np.arange(numModelFrame)):
            animalFrame[:,:,i] = cv.cvtColor(cv.subtract(storeFrame[:,:,:,i], imageMedian),cv.COLOR_RGB2GRAY)

        # Calculate Threshold value
        self.threshold = np.mean(animalFrame) + 2 * np.std(animalFrame) # 2 sigma over background as threshold

        for i in np.arange(numModelFrame):
            animalSize[i] = np.sum(animalFrame[:,:,i] > self.threshold)

        self.animalSize = np.mean(animalSize)

        print('ROI_image_stream : Training Backgroundsubtractor...')
        self.backSub.apply(imageMedian,learningRate=1)
        # TODO : train image backward! (reversed)
        if not self.stableBackground:
            for i in np.arange(numModelFrame):
                self.backSub.apply(storeFrame[:,:,:,i], learningRate=0.01)

        print('ROI_image_stream : Done')

        self.isBackgroundSubtractorTrained = True

    def getFrame(self, frame_number):
        """
        getFrame : return original frame
        ----------------------------------------------------------------
        frame_number : int : frame to process
        ----------------------------------------------------------------
        returns frame 
        """
        self.vc.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, image = self.vc.read()
        #image correction can be applied. but this slows the process a lot.
        #ex) image = np.clip(((self.alpha * image + self.beta) / 255) ** self.gamma * 255, 0, 255).astype(np.uint8)
        if not ret:
            raise(BaseException(f'ROI_image_stream : Can not retrieve frame # {frame_number}'))
        return image

    def drawFrame(self, frame_number):
        """
        drawFrame : draw original frame
        --------------------------------------------------------------------------------
        frame_number : int : frame to process
        """
        image = self.getFrame(frame_number)

        cv.putText(image,f'Frame : {frame_number:.0f}', [0, int(image.shape[0] - 1)], fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8,
                   color=[255, 255, 255], thickness=1)
        cv.imshow('Frame', image)
        cv.waitKey()

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
        self.vcIOthread = Thread(target=self.__readVideo, args=())
        self.vcIOthread.daemon = True # indicate helper thread
        self.frame_number_array_idx = 0
        if not self.vcIOthread.isAlive():
            self.vcIOthread.start()

        # ROI detection Thread and Queue
        self.blobQ = Queue(maxsize=200)
        self.roiDetectionThread = Thread(target=self.__processFrames, args=())
        self.roiDetectionThread.daemon = True
        self.roiDetectionThread.start()

        self.isMultithreading = True

    def __getFrameSize(self, vc)
        """
        __getFrameSize : get frame size from the VideoCapture object.
            I can not trust vid.get(cv.CAP_PROP_FRAME_COUNT), because sometime I can't retrieve the last frame with vid.read()
        """
        num_frame = int(self.vc.get(cv.CAP_PROP_FRAME_COUNT))
        self.vc.set(cv.CAP_PROP_POS_FRAMES, self.num_frame-1)
        ret, _ = self.vc.read()
        while not ret:
            print(f'ROI_image_stream : Can not read the frame from the last position. Decreasing the total frame count')
            num_frame -= 1
            self.vc.set(cv.CAP_PROP_POS_FRAMES, self.num_frame)
            ret, _ = self.vc.read()
        return num_frame

    def __readVideoPath(self, path_data)
    """
    parse the path of the video. if not found or multiple files are found, evoke an error
    """
        if path_data.is_file():
            if path_data.suffix in ['.mkv', '.avi', '.mp4']: # path is video.mkv
                return path_data
            else:
                raise(BaseException(f'ROI_image_stream : Following file is not a supported video type : {path_data.suffix}'))
        elif path_data.is_dir():
            vidlist = []
            vidlist.extend([i for i in path_data.glob('*.mkv')])
            vidlist.extend([i for i in path_data.glob('*.avi')])
            vidlist.extend([i for i in path_data.glob('*.mp4')])
            if len(vidlist) == 0:
                raise(BaseException(f'ROI_image_stream : Can not find video in {path_data}'))
            elif len(vidlist) > 1:
                raise(BaseException(f'ROI_image_stream : Multiple video files found in {path_data}'))
            else:
                return vidlist[0]
        else:
            raise(BaseException(f'ROI_image_stream : Can not find video file in {path_data}'))

    def __readVideo(self):
        """
        __readVideo : multithreading. read video, extract frame listed in self.frame_number_array and store in self.frameQ
        """
        print('ROI_image_stream : Video IO Thread started\n')
        while True:
            if not self.frameQ.full():
                frame_number = self.frame_number_array[self.frame_number_array_idx]
                image = self.getFrame(frame_number)
                self.frameQ.put((frame_number, image))
                self.frame_number_array_idx += 1
            else:
                time.sleep(0.1)
            if self.frame_number_array_idx >= self.frame_number_array.shape[0]:
                break
        print('ROI_image_stream : Video IO Thread stopped\n')

    def __processFrames(self):
        """
        __processFrames : multithreading. extract ROI from frame stored in self.frameQ and store in self.blobQ
        """
        print('ROI_image_stream : ROI extraction Thread started\n')
        # run until frameQ is empty and thread is dead 
        while not(self.frameQ.empty()) or self.vcIOthread.isAlive():
            if not self.blobQ.full():
                frame_number, image = self.frameQ.get()
                detected_blob = self.__findBlob(image)
                self.blobQ.put((frame_number, image, detected_blob))
            else:
                time.sleep(0.1)
        print('ROI_image_stream : ROI extraction Thread stopped\n')

    def getKernel(self, size):
        size = int(size)
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size),
                                        ((int((size - 1) / 2), int((size - 1) / 2))))
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
        image = cv.bitwise_and(image, image, mask=self.global_mask)
        
        # Background Subtractor : adaptive learning rate adjustment
        normal_lr = 1e-4
        stationary_lr = 1e-6
        stationary_criterion = self.animalSize * 1.1

        masked_image = self.backSub.apply(image, learningRate=self.backSub_lr)
        if self.stableBackground: # if stable, don't change the background model
            self.backSub_lr = 1e-8
        else:
            if (self.masked_image == []): # empty
                self.backSub_lr = normal_lr
            else:
                if cv.countNonZero(cv.bitwise_and(masked_image, self.masked_image)) > stationary_criterion:
                    self.backSub_lr = stationary_lr
                else:
                    self.backSub_lr = normal_lr
        self.masked_image = masked_image

        # opening -> delete noise : erode and dilate
        # closing -> make into big object : dilate and erode
        denoised_mask = masked_image
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, self.getKernel(3))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, self.getKernel(5))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_CLOSE, self.getKernel(10))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, self.getKernel(7))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, self.getKernel(12))

        # detect
        detected_blob = self.detector.detect(denoised_mask)

        if len(detected_blob) == 0: # if not found, try softer one
            detected_blob = self.detector_soft.detect(denoised_mask)

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
            image = self.getFrame(frame_number)
            detected_blob = self.__findBlob(image)

        else: # frame number is not provided
            if not self.isMultithreading: # if multithreading is not used
                raise(TypeError('getROIImage() missing 1 required positional argument: \'frame_number\'\n'
                                'If you are trying to use this function as multithreading, check if you called startROIextractionThread()'))
            frame_number, image, detected_blob = self.blobQ.get()

        # Further process detected blobs
        if len(detected_blob) == 0 :
            raise(BlobDetectionFailureError('No Blob'))
        elif len(detected_blob) > 1:# if multiple blob is detected, select the largest one
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
