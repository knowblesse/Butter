from collections import deque
from queue import Queue
from threading import Thread
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import norm
from tqdm import tqdm
import warnings

class ROI_image_stream():
    def __init__(self,path_data, ROI_size, setMask=True):
        """
        __init__ : initialize ROI image extraction stream object
        ****************************************************************
        path_data : Path object : path of the video or a folder containing the video
        ROI_size : weight/height of the extracting ROI. This must be an even number.
        setMask : if True, ROI selection screen appears.
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

        # Set Mask
        if setMask:
            mask_position = cv.selectROI('Select ROI', self.getFrame(0))
            cv.destroyWindow('Select ROI')
            self.global_mask = np.zeros(self.frame_size, dtype=np.uint8)
            self.global_mask[mask_position[1]:mask_position[1]+mask_position[3], mask_position[0]:mask_position[0]+mask_position[2]] = 255
        else:
            self.global_mask = 255 * np.ones(self.frame_size, dtype=np.uint8)

    def buildForegroundModel(self, num_frames2use = 200, verbose=False):
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
        self.foregroundModel = frameStorage

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

        noContourFoundIndex = []
        for i in tqdm(np.arange(num_frames2use)):
            image = cv.cvtColor(cv.absdiff(frameStorage[:,:,:,i], self.medianFrame), cv.COLOR_RGB2GRAY)
            animalThreshold[i] = np.quantile(image, 0.99) # consider only the top 1% of the intensity as the foreground object.
            binaryImage = cv.threshold(image,animalThreshold[i], 255, cv.THRESH_BINARY)[1]
            denoisedBinaryImage = self.__denoiseBinaryImage(binaryImage)
            # Find the largest contour
            cnts = cv.findContours(denoisedBinaryImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
            maxCntSize = 0
            maxCntIndex = None
            for j, cnt in enumerate(cnts):
                area = cv.contourArea(cnt)
                if area > maxCntSize:
                    maxCntSize = area
                    maxCntIndex = j
            # if no contour is found, skip
            if len(cnts) == 0:
                noContourFoundIndex.append(i)
                continue
            # Draw contours to the initial foreground model
            self.foregroundModel[:,:,:,i] = cv.drawContours(self.foregroundModel[:,:,:,i].astype(np.uint8), cnts, maxCntIndex, (255,0,0))

            # Calculate feature information from the largest contour
            area = cv.contourArea(cnts[maxCntIndex])
            perimeter = cv.arcLength(cnts[maxCntIndex], closed=True)
            animalSize[i] = area
            animalConvexity[i] = area / cv.contourArea(cv.convexHull(cnts[maxCntIndex]))
            animalCircularity[i] = 4 * np.pi * area / (perimeter ** 2)

        # Delete frames where no contour is found
        if len(noContourFoundIndex) != 0:
            animalThreshold = np.delete(animalThreshold, noContourFoundIndex)
            animalSize = np.delete(animalSize, noContourFoundIndex)
            animalConvexity = np.delete(animalConvexity, noContourFoundIndex)
            animalCircularity = np.delete(animalCircularity, noContourFoundIndex)

        self.animalThreshold = np.median(animalThreshold)
        self.animalSize = {'median': np.median(animalSize), 'sd': np.std(animalSize)}
        self.p2pDisplacement = {'median' : 0.33 * np.sqrt(self.animalSize['median']), 'sd': 0.33 * np.sqrt(self.animalSize['median'])} # see log.txt 22MAR21
        self.animalConvexity = {'median': np.median(animalConvexity), 'sd': np.std(animalConvexity)}
        self.animalCircularity = {'median': np.median(animalCircularity), 'sd': np.std(animalCircularity)}

        if verbose:
            print(f'Foreground Model built.')
            print(f'Animal Size : {self.animalSize["median"]:.2f} ({self.animalSize["sd"]:.2f})')
            print(f'Animal Convexity : {self.animalConvexity["median"]:.2f} ({self.animalConvexity["sd"]:.2f})')
            print(f'Animal Circularity : {self.animalCircularity["median"]:.2f} ({self.animalCircularity["sd"]:.2f})')
        self.pastFrameImage = self.getFrame(0)

        self.isForegroundModelBuilt = True

    def getFrame(self, frame_number, applyGlobalMask=False):
        """
        getFrame : return original frame
        -------------------------------------------------------------------------------------
        frame_number : int : frame to process
        applyGlobalMask : bool : if true, the global mask, set from the beginning, is applied
        -------------------------------------------------------------------------------------
        returns frame 
        """
        self.vc.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, image = self.vc.read()
        if not ret:
            raise(BaseException(f'ROI_image_stream : Can not retrieve frame # {frame_number}'))
        if applyGlobalMask:
            image = cv.bitwise_and(image, image, mask=self.global_mask)
        return image


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
        self.prevPoint = None
        self.roiDetectionThread = Thread(target=self.__processFrames, args=())
        self.roiDetectionThread.daemon = True
        self.roiDetectionThread.start()

        self.isMultithreading = True

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
                detected_blob = self.__findBlob(image, prevPoint=self.prevPoint)
                self.blobQ.put((frame_number, image, detected_blob))
                if detected_blob is not None:
                    self.prevPoint = detected_blob
            else:
                time.sleep(0.1)
        print('ROI_image_stream : ROI extraction Thread stopped\n')

    def __findBlob(self, image, prevPoint=None):
        """
        __findBlob: from given image, apply noise filter and find blob.
        --------------------------------------------------------------------------------
        image : 3D np.array : image to process
        --------------------------------------------------------------------------------
        return list of blobs
        --------------------------------------------------------------------------------
        """
        # Constant
        maxBlob = 3 # how many maximum size blobs to find 
        # Add current image to the pastFrames Storage
        if prevPoint is not None:
            self.pastFrameImage = cv.addWeighted(self.pastFrameImage, 0.9, image, 0.1, 0)

        # Foreground Detector
        weight_mediandiff = 0.8
        weight_recentdiff = 1 - weight_mediandiff

        if weight_recentdiff == 0 :
            image = cv.absdiff(image, self.medianFrame)
        else:
            image = cv.addWeighted(
                cv.absdiff(image, self.medianFrame), weight_mediandiff,
                cv.absdiff(image, self.pastFrameImage), weight_recentdiff,
                0)

        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        binaryImage = cv.threshold(image, self.animalThreshold, 255, cv.THRESH_BINARY)[1]

        denoisedBinaryImage = self.__denoiseBinaryImage(binaryImage)

        # Find the largest contour
        cnts = cv.findContours(denoisedBinaryImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        largestContourIndex = np.argsort(np.array([cv.contourArea(cnt) for cnt in cnts]))[-1:(-1-maxBlob):-1]
        largestContours = [cnts[i] for i in largestContourIndex]

        # Calculate Feature information
        centers = np.array([np.round(cv.minEnclosingCircle(cnt)[0]).astype(int) for cnt in largestContours])
        area = np.array([cv.contourArea(cnt) for cnt in largestContours])
        perimeter = np.array([cv.arcLength(cnt, closed=True) for cnt in largestContours])

        animalSize = area
        animalConvexity = area / np.array([cv.contourArea(cv.convexHull(cnt)) for cnt in largestContours])
        animalCircularity = 4 * np.pi * area / (perimeter ** 2)

        L_Size = np.max([
            norm.cdf(animalSize + self.animalSize['sd'] * 0.1, self.animalSize['median'], self.animalSize['sd'])
            - norm.cdf(animalSize - self.animalSize['sd'] * 0.1, self.animalSize['median'], self.animalSize['sd']),
            1e-10*np.ones(animalSize.shape)], axis=0)
        L_Convexity = np.max([
            norm.cdf(animalConvexity + self.animalConvexity['sd'] * 0.1, self.animalConvexity['median'],self.animalConvexity['sd'])
            - norm.cdf(animalConvexity - self.animalConvexity['sd'] * 0.1, self.animalConvexity['median'],self.animalConvexity['sd']),
            1e-10*np.ones(animalConvexity.shape)], axis=0)
        L_Circularity = np.max([
            norm.cdf(animalCircularity + self.animalCircularity['sd'] * 0.1, self.animalCircularity['median'],self.animalCircularity['sd'])
            - norm.cdf(animalCircularity - self.animalCircularity['sd'] * 0.1, self.animalCircularity['median'],self.animalCircularity['sd']),
            1e-10*np.ones(animalCircularity.shape)], axis=0)

        likelihoods = np.log(L_Size) + np.log(L_Convexity) + np.log(L_Circularity)

        # If no blob is found, then return None as output
        if len(likelihoods) == 0:
            return None

        # If prevPoint is provided, then use previous location to calculate more accurate loglikelihood
        if prevPoint is not None:
            distance = np.sum((prevPoint - centers)**2, axis=1)**0.5
            L_Distance = np.max([
                norm.cdf(distance + self.p2pDisplacement['sd'] * 0.1, self.p2pDisplacement['median'],
                         self.p2pDisplacement['sd'])
                - norm.cdf(distance - self.p2pDisplacement['sd'] * 0.1, self.p2pDisplacement['median'],
                           self.p2pDisplacement['sd']),
                1e-10 * np.ones(distance.shape)], axis=0)  # 14.530208395578228, 13.609712279904377
            L_Distance = 0.1 * L_Distance # penalty term for fixing to a wrong location
            likelihoods += np.log(L_Distance)

        return centers[np.argmax(likelihoods)]

    def drawROI(self, frame_number):
        img = self.getFrame(frame_number)
        ROIcenter = self.getROIImage(frame_number)[1]
        cv.rectangle(img, [ROIcenter[1]-self.half_ROI_size, ROIcenter[0]-self.half_ROI_size], [ROIcenter[1]+self.half_ROI_size, ROIcenter[0]+self.half_ROI_size], (255,0,0), 3)
        cv.imshow('Frame', img)
        cv.waitKey()

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

    def getKernel(self, size):
        size = int(size)
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size),
                                        ((int((size - 1) / 2), int((size - 1) / 2))))

    def __denoiseBinaryImage(self, binaryImage):
        """
        __denoiseBinaryImage : remove small artifacts from the binary Image using the cv.morphologyEx function
        :param binaryImage: boolean image
        :return: denoised binary image
        """
        # opening -> delete noise : erode and dilate
        # closing -> make into big object : dilate and erode
        denoisedBinaryImage = binaryImage
        denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_OPEN, self.getKernel(3))
        denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_OPEN, self.getKernel(5))
        denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_CLOSE, self.getKernel(10))
        # denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_OPEN, self.getKernel(7))
        # denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_OPEN, self.getKernel(12))
        return denoisedBinaryImage

    def __readVideoPath(self, path_data):
        """
        parse the path of the video. if not found or multiple files are found, evoke an error
        """
        if path_data.is_file():
            if path_data.suffix in ['.mkv', '.avi', '.mp4', '.mpg']: # path is video.mkv
                return path_data
            else:
                raise(BaseException(f'ROI_image_stream : Following file is not a supported video type : {path_data.suffix}'))
        elif path_data.is_dir():
            vidlist = []
            vidlist.extend([i for i in path_data.glob('*.mkv')])
            vidlist.extend([i for i in path_data.glob('*.avi')])
            vidlist.extend([i for i in path_data.glob('*.mp4')])
            vidlist.extend([i for i in path_data.glob('*.mpg')])
            if len(vidlist) == 0:
                raise(BaseException(f'ROI_image_stream : Can not find video in {path_data}'))
            elif len(vidlist) > 1:
                raise(BaseException(f'ROI_image_stream : Multiple video files found in {path_data}'))
            else:
                return vidlist[0]
        else:
            raise(BaseException(f'ROI_image_stream : Can not find video file in {path_data}'))

    def getROIImage(self, frame_number=-1):
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

        if detected_blob is None:
            raise(BlobDetectionFailureError('No Blob'))

        blob_center_row = detected_blob[1]
        blob_center_col = detected_blob[0]

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
    # diagonal line
    l = ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5
    # temporal degree value
    temp_deg = np.rad2deg(np.arccos((c2 - c1) / l))
    # if r1 <= r2, then [0, 180) degree = temp_deg
    # if r1 > r2, then [180. 360) degree = 360 - temp_deg
    deg = 360 * np.array(r1 > r2, dtype=int) + (np.array(r1 <= r2, dtype=int) - np.array(r1 > r2, dtype=int)) * temp_deg
    return np.round(deg).astype(np.int)
