import cv2 as cv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from threading import Thread
from queue import Queue
import time
from collections import deque

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
        self.num_frame = self.getFrameCount()

        # stream status 
        self.isForegroundModelBuilt = False
        self.isMultithreading = False

        # ROI size
        self.ROI_size = ROI_size
        self.half_ROI_size = int(np.round(self.ROI_size/2))
        if ROI_size%2 != 0 :
            raise(BaseException('ROI_size is not dividable with 2!'))

        # Foreground detector
        self.pastFrameNumber = 100
        self.pastFrames = deque([],maxlen=self.pastFrameNumber)

        # Set Mask
        if setMask:
            mask_position = cv.selectROI('Select ROI', self.getFrame(0))
            cv.destroyWindow('Select ROI')
            self.global_mask = np.zeros(self.frame_size, dtype=np.uint8)
            self.global_mask[mask_position[1]:mask_position[1]+mask_position[3], mask_position[0]:mask_position[0]+mask_position[2]] = 255
        else:
            self.global_mask = 255 * np.ones(self.frame_size, dtype=np.uint8)

    def buildForegroundModel(self, num_frames2use = 200):
        """
        buildForegroundModel : build a foreground model for initial movement detection
        ----------------------------------------------------------------------------------------
        num_frames2use : int : number of frames to use for building the foreground model
        """
        # Read and Store Multiple Frames and Extract Initial image for background model
        print('ROI_image_stream : Acquiring frames to analyze')
        frameStorage = np.zeros((self.frame_size[0], self.frame_size[1], 3, num_frames2use), dtype=np.uint8)
        for i, frame in enumerate(tqdm(np.round(np.linspace(0, self.num_frame-1, num_frames2use)).astype(int))):
            image = self.getFrame(frame, applyGlobalMask=True)
            frameStorage[:,:,:,i] = image
        self.medianFrame = np.median(frameStorage,axis=3).astype(np.uint8)

        # Build a foreground Model
        """
        Run through sufficient number of frames to calculate proper size and the threshold of the foreground model.
        Later, the value computed from this psudo-foreground object will be used to detect the foreground object.
        """
        print('ROI_image_stream : Initial Foreground Model building...')

        animalSize = np.zeros(num_frames2use)
        animalThreshold = np.zeros(num_frames2use)
        animalConvexity = np.zeros(num_frames2use)
        animalCircularity = np.zeros(num_frames2use)

        for i in tqdm(np.arange(num_frames2use)):
            image = cv.cvtColor(cv.absdiff(frameStorage[:,:,:,i], self.medianFrame), cv.COLOR_RGB2GRAY)
            animalThreshold[i] = np.quantile(image, 0.99) # consider only the top 1% of the intensity as the forground object.
            binaray_image = cv.threshold(image,animalThreshold[i], 255, cv.THRESH_BINARY)[1]

            # Find the largest contour
            cnts = cv.findContours(binaray_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
            maxCntSize = 0
            maxCntIndex = None
            for j, cnt in enumerate(cnts):
                area = cv.contourArea(cnt)
                if area > maxCntSize:
                    maxCntSize = area
                    maxCntIndex = j

            # Calculate feature information from the largest contour
            area = cv.contourArea(cnts[maxCntIndex])
            perimeter = cv.arcLength(cnts[maxCntIndex], closed=True)
            animalSize[i] = area
            animalConvexity[i] = area / cv.contourArea(cv.convexHull(cnts[maxCntIndex]))
            animalCircularity[i] = 4 * np.pi * area / (perimeter ** 2)

        self.animalThreshold = np.median(animalThreshold)
        self.animalSize = {'median': np.median(animalSize), 'sd': np.std(animalSize)}
        self.animalConvexity = {'median': np.median(animalConvexity), 'sd': np.std(animalConvexity)}
        self.animalCircularity = {'median': np.median(animalCircularity), 'sd': np.std(animalCircularity)}

        self.isForegroundModelBuilt = True

    def getFrame(self, frame_number, applyGlobalMask=False):
        """
        getFrame : return original frame
        ----------------------------------------------------------------
        frame_number : int : frame to process
        ----------------------------------------------------------------
        returns frame 
        """
        self.vc.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, image = self.vc.read()
        if not ret:
            raise(BaseException(f'ROI_image_stream : Can not retrieve frame # {frame_number}'))
        if applyGlobalMask:
            image = cv.bitwise_and(image, image, mask=self.global_mask)
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
            Processed ROI is stored in self.blobQ
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

    def getFrameCount(self):
        """
        getFrameCount : get frame size from the VideoCapture object.
            I can not trust vid.get(cv.CAP_PROP_FRAME_COUNT), because sometime I can't retrieve the last frame with vid.read()
        """
        num_frame = int(self.vc.get(cv.CAP_PROP_FRAME_COUNT))
        self.vc.set(cv.CAP_PROP_POS_FRAMES, num_frame-1)
        ret, _ = self.vc.read()
        while not ret:
            num_frame -= 1
            self.vc.set(cv.CAP_PROP_POS_FRAMES, num_frame)
            ret, _ = self.vc.read()
        return num_frame

    def getFPS(self):
        return self.vc.get(cv.CAP_PROP_FPS)

    def __readVideoPath(self, path_data):
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
                image = self.getFrame(frame_number, applyGlobalMask=True)
                self.frameQ.put((frame_number, image))
                self.frame_number_array_idx += 1
            else:
                time.sleep(0.1)
            # Check if all frames are added.
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
                detected_blobs = self.__findBlob(image)
                self.blobQ.put((frame_number, image, detected_blobs))
            else:
                time.sleep(0.1)
        print('ROI_image_stream : ROI extraction Thread stopped\n')

    def getKernel(self, size):
        size = int(size)
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size),
                                        ((int((size - 1) / 2), int((size - 1) / 2))))
    def __findBlob(self, image, maxBlob = 3, sequentialCalling=False):
        """
        __findBlob: from given image, apply noise filter and find blob.
        --------------------------------------------------------------------------------
        image : 3D np.array : image to process
        maxBlob : int : maximum number of blobs to return 
        sequentialCalling : bool : set true if this function is called sequentially through time.
            Every image called with this function is added to the self.pastFrames
        --------------------------------------------------------------------------------
        return list of blobs
        --------------------------------------------------------------------------------
        """
        # Add current image to the pastFrames Storage
        if sequentialCalling:
            self.pastFrames.append(image)

        # Foreground Detector
        weight_mediandiff = (0.5 + 0.5*(self.pastFrameNumber - len(self.pastFrames))/self.pastFrameNumber)
        weight_recentdiff = 1 - weight_mediandiff

        if weight_recentdiff == 0 :
            image = cv.absdiff(image, self.medianFrame)
        else:
            image = cv.addWeighted(
                cv.absdiff(image, self.medianFrame), weight_mediandiff,
                cv.absdiff(image, np.median(self.pastFrames)), weight_recentdiff,
                0)

        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        binaray_image = cv.threshold(image, self.animalThreshold, 255, cv.THRESH_BINARY)[1]

        # opening -> delete noise : erode and dilate
        # closing -> make into big object : dilate and erode
        denoised_mask = binaray_image
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, self.getKernel(3))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, self.getKernel(5))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_CLOSE, self.getKernel(10))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, self.getKernel(7))
        denoised_mask = cv.morphologyEx(denoised_mask, cv.MORPH_OPEN, self.getKernel(12))

        # Find the largest contour
        cnts = cv.findContours(denoised_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        largestContourIndex = np.argsort(np.array([cv.contourArea(cnt) for cnt in cnts]))[-1:(-1-maxBlob):-1]
        largestContours = [cnts[i] for i in largestContourIndex]

        # Calculate Feature information
        area = np.array([cv.contourArea(cnt) for cnt in largeestContours])
        perimeter = np.array([cv.arcLength(cnt, closed=True) for cnt in largeestContours])

        animalSize = area
        animalConvexity = area / np.array([cv.contourArea(cv.convexHull(cnt)) for cnt in largeestContours])
        animalCircularity = 4 * np.pi * area / (perimeter ** 2)

        likelihoods = np.log(norm.pdf(animalSize, self.animalSize['median'], self.animalSize['sd'])) +\
                     np.log(norm.pdf(animalConvexity, self.animalConvexity['median'], self.animalConvexity['sd'])) +\
                     np.log(norm.pdf(animalCircularity, self.animalCircularity['median'], self.animalCircularity['sd']))

        # output center
        detected_blobs = [(cv.minEnclosingCircle(cnt)[0], likelihood) for cnt, likelihood in zip(largestContours,likelihoods)]
        

        return detected_blobs

    def drawROI(self, frame_number):
        img = self.getFrame(frame_number)
        ROIcenter = self.getROIImage(frame_number)[1]
        cv.rectangle(img, [ROIcenter[0]-self.half_ROI_size, ROIcenter[1]-self.half_ROI_size], [ROIcenter[0]+self.half_ROI_size, ROIcenter[1]+self.half_ROI_size])
        return img

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
        if not self.isForegroundModelBuilt:
            raise(BaseException('BackgroundSubtractor is not trained'))

        if frame_number != -1: # frame_number is provided
            image = self.getFrame(frame_number, applyGlobalMask=True)
            detected_blob = self.__findBlob(image)

        else: # frame number is not provided
            if not self.isMultithreading: # if multithreading is not used
                raise(TypeError('getROIImage() missing 1 required positional argument: \'frame_number\'\n'
                                'If you are trying to use this function as multithreading, check if you called startROIextractionThread()'))
            frame_number, image, detected_blob = self.blobQ.get()

        # TODO now this is not a blob. rather, contours

        # Further process detected blobs

        blob_center_row, blob_center_col = int(np.round(detected_blob[1])) , int(np.round(detected_blob[0]))

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
