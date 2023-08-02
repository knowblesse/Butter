import queue
from queue import Queue
from threading import Thread
import time
import cv2 as cv
import numpy as np
from pathlib import Path
from scipy.stats import norm
from tqdm import tqdm

class ROI_image_stream():
    def __init__(self,path_data, ROI_size):
        """
        __init__ : initialize ROI image extraction stream object
        ****************************************************************
        path_data : Path object : path of the video or a folder containing the video
        ROI_size : width/height of the extracting ROI. This must be an even number.
        setMask : if True, ROI selection screen appears.
        """
        # Setup VideoCapture
        self.path_video = self.__readVideoPath(path_data)
        self.vc = cv.VideoCapture(str(self.path_video))
        self.frame_size = (int(self.vc.get(cv.CAP_PROP_FRAME_HEIGHT)), int(self.vc.get(cv.CAP_PROP_FRAME_WIDTH)))
        self.num_frame = int(self.vc.get(cv.CAP_PROP_FRAME_COUNT))
        self.cur_header = 0
        self.fps = int(self.vc.get(cv.CAP_PROP_FPS))

        # stream status 
        self.sampleFrames = []
        self.isForegroundModelBuilt = False
        self.isBackgroundModelBuilt = False
        self.isMultithreading = False

        # ROI size
        self.ROI_size = ROI_size
        self.half_ROI_size = int(np.round(self.ROI_size/2))
        if ROI_size%2 != 0 :
            raise(BaseException('ROI_size is not dividable with 2!'))

        # Set Mask
        self.global_mask = 255 * np.ones(self.frame_size, dtype=np.uint8)

    def setGlobalMask(self, mask=[]):
        if not mask:
            mask_position = cv.selectROI('Select ROI', self.getFrame(0, applyGlobalMask=False))
            cv.destroyWindow('Select ROI')
            self.global_mask = np.zeros(self.frame_size, dtype=np.uint8)
            self.global_mask[mask_position[1]:mask_position[1] + mask_position[3],
            mask_position[0]:mask_position[0] + mask_position[2]] = 255
        else:
            self.global_mask = mask
    def getGlobalMask(self):
        return self.global_mask

    def setForegroundModel(self, foregroundModel):
        """
        setForegroundModel : Manually set foreground model
        """
        self.foregroundModel = foregroundModel
        self.pastFrameImage = self.getFrame(0)
        self.isForegroundModelBuilt = True

    def setBackgroundModel(self, background):
        self.background = background
        self.isBackgroundModelBuilt = True

    def saveSampleFrames(self, num_frames2use=200):
        """
        saveSampleFrames : Read and Store Multiple Frames and Extract Initial image for background model
        """
        # Check global mask
        if np.all(self.global_mask == 255 * np.ones(self.frame_size, dtype=np.uint8)): # default global mask
            print('ROI_image_stream : Warning. Currently using the default global mask')

        print('ROI_image_stream : Acquiring frames to analyze')
        self.sampleFrames = np.zeros((self.frame_size[0], self.frame_size[1], 3, num_frames2use), dtype=np.uint8)
        self._rewindPlayHeader()
        for i, frame_number in enumerate(
                tqdm(np.round(np.linspace(0, self.num_frame - 1, num_frames2use)).astype(int))):
            while self.cur_header <= frame_number:
                if self.cur_header < frame_number: # Touch Frame
                    ret = self.vc.grab()
                else: # Read Frame
                    ret, frame = self.vc.read()

                if (not ret) and self.cur_header > self.num_frame-10:
                    # if the header is nearly the end of the video and failed to retrieve a frame,
                    # then the frame number calculated from `set(cv.CAP_PROP_FRAME_COUNT)` is wrong
                    self.num_frame = self.cur_header
                    print(f'ROI_image_stream : total frame number is wrong. Calibrated to {self.num_frame}')
                elif not ret:
                    raise(BaseException(f'ROI_image_stream : Corrupted video file. Can not get frame from {self.cur_header}'))
                self.cur_header += 1
            # Save
            frame = cv.bitwise_and(frame, frame, mask=self.global_mask)
            self.sampleFrames[:, :, :, i] = frame

    def buildBackgroundModel(self):
        """
        buildBackgroundModel : build a background model
        ----------------------------------------------------------------------------------------
        """
        # Check if sampel frames exist
        if not len(self.sampleFrames):
            print(f'ROI_image_stream : Sample frame does not exist. Run saveSampleFrames() to acquire frames')
            return
        self.background = np.median(self.sampleFrames, axis=3).astype(np.uint8)
        self.isBackgroundModelBuilt = True

    def buildForegroundModel(self, verbose=False):
        """
        buildForegroundModel : build a foreground model for initial movement detection
        Run through sufficient number of frames to calculate proper size and the threshold of the foreground model.
        Later, the value computed from this psudo-foreground object will be used to detect the foreground object.
        ----------------------------------------------------------------------------------------
        """
        # Check if sampel frames exist
        if not len(self.sampleFrames):
            print(f'ROI_image_stream : Sample frame does not exist. Run saveSampleFrames() to acquire frames')
            return

        # Variable to save
        numSampleFrame = self.sampleFrames.shape[3]
        animalSize = np.zeros(numSampleFrame)
        animalThreshold = np.zeros(numSampleFrame)
        animalConvexity = np.zeros(numSampleFrame)
        animalCircularity = np.zeros(numSampleFrame)

        noContourFoundIndex = []
        for i in tqdm(np.arange(numSampleFrame)):
            image = cv.cvtColor(cv.absdiff(self.sampleFrames[:,:,:,i], self.background), cv.COLOR_RGB2GRAY)
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
            self.sampleFrames[:,:,:,i] = cv.drawContours(self.sampleFrames[:,:,:,i].astype(np.uint8), cnts, maxCntIndex, (255,0,0))

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

        foregroundModel = {
                'animalThreshold': np.median(animalThreshold),
                'animalSize': {'median': np.median(animalSize), 'sd': np.std(animalSize)},
                'p2pDisplacement': {'median' : 0.33 * np.sqrt(np.median(animalSize)), 'sd': 0.33 * np.sqrt(np.median(animalSize))}, # see log.txt 22MAR21
                'animalConvexity': {'median': np.median(animalConvexity), 'sd': np.std(animalConvexity)},
                'animalCircularity': {'median': np.median(animalCircularity), 'sd': np.std(animalCircularity)}
                }

        if verbose:
            print(f"Foreground Model built.")
            print(f"Animal Threshold : {foregroundModel['animalThreshold']:.2f}")
            print(f"Animal p2p Displacement : {foregroundModel['p2pDisplacement']['median']:.2f} ({foregroundModel['p2pDisplacement']['sd']:.2f})")
            print(f"Animal Size : {foregroundModel['animalSize']['median']:.2f} ({foregroundModel['animalSize']['sd']:.2f})")
            print(f"Animal Convexity : {foregroundModel['animalConvexity']['median']:.2f} ({foregroundModel['animalConvexity']['sd']:.2f})")
            print(f"Animal Circularity : {foregroundModel['animalCircularity']['median']:.2f} ({foregroundModel['animalCircularity']['sd']:.2f})")
        self.pastFrameImage = self.getFrame(0) 
        self.foregroundModel = foregroundModel
        self.isForegroundModelBuilt = True 
 
    def getFrame(self, frame_number, applyGlobalMask = True):
        """
        getFrame : return original frame
        -------------------------------------------------------------------------------------
        frame_number : int : frame to process
        *Notice : this code, which look through the whole video from the beginning looks very
        inefficient, but it guarantee the retrieved frame is accurate. See opencv's Issue #9053
        -------------------------------------------------------------------------------------
        returns frame 
        """
        # check if the input is int
        if type(frame_number) is not int:
            raise(BaseException('ROI_image_stream : frame_number must be an integer'))
        self._rewindPlayHeader()
        image = None
        while self.cur_header != frame_number+1:
            ret, image = self.vc.read()
            if not ret:
                raise(BaseException(f'ROI_image_stream : Can not retrieve frame # {frame_number}'))
            else:
                self.cur_header += 1
        if applyGlobalMask:
            image = cv.bitwise_and(image, image, mask=self.global_mask)
        return image

    def startROIextractionThread(self, start_frame, stride=5):
        """
        startROIextractionThread : start ROI extraction Thread for continuous processing.
            When called, two thread (video read and opencv ROI detection) is initiated.
            Processed ROI is stored in self.blobQ
        --------------------------------------------------------------------------------
        stride : integer : The function read one frame from from every (stride) number of frames
        """
        # Video IO Thread and Queue
        self.frameQ = Queue(maxsize=200)
        self.vcIOthread = Thread(target=self.__readVideo, args=(start_frame, stride,))
        self.vcIOthread.daemon = True # indicate helper thread
        if not self.vcIOthread.is_alive():
            self.vcIOthread.start()

        # ROI detection Thread and Queue
        self.blobQ = Queue(maxsize=200)
        self.prevPoint = None
        self.roiDetectionThread = Thread(target=self.__processFrames, args=())
        self.roiDetectionThread.daemon = True
        self.roiDetectionThread.start()

        self.isMultithreading = True

    def drawROI(self, frame_number):
        """
        [DEBUG]
        """
        img = self.getFrame(frame_number)
        ROIcenter = self.getROIImage(frame_number)[1]
        cv.rectangle(img, [ROIcenter[1]-self.half_ROI_size, ROIcenter[0]-self.half_ROI_size], [ROIcenter[1]+self.half_ROI_size, ROIcenter[0]+self.half_ROI_size], (255,0,0), 3)
        cv.imshow('Frame', img)
        cv.waitKey()

    def drawFrame(self, frame_number):
        """
        [DEBUG]
        drawFrame : draw original frame
        --------------------------------------------------------------------------------
        frame_number : int : frame to process
        """
        image = self.getFrame(frame_number)

        cv.putText(image,f'Frame : {frame_number:.0f}', [0, int(image.shape[0] - 1)], fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8,
                   color=[255, 255, 255], thickness=1)
        cv.imshow('Frame', image)
        cv.waitKey()

    def getROIImage(self, frame_number=-1):
        """
        getROIImage : run __findBlob(image)
        -------------------------------------------------------------------------------- 
        frame_number : int or int numpy array : frame to process
        ---------------------------------------------------------------- 
        chosen_image : 3D numpy array : ROI frame  
        blob_center : (r,c) : location of the center of the blob
        """
        if (not self.isForegroundModelBuilt) or (not self.isBackgroundModelBuilt):
            raise(BaseException('BackgroundSubtractor is not trained'))

        if frame_number != -1: # frame_number is provided
            image = self.getFrame(frame_number)
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

    def _rewindPlayHeader(self):
        """
        _rewindPlayHeader : rewind the play header of the VideoCapture object
        """
        self.vc.set(cv.CAP_PROP_POS_FRAMES, 0)
        if self.vc.get(cv.CAP_PROP_POS_FRAMES) != 0:
            raise(BaseException('ROI_image_stream : Can not set the play header to the beginning'))
        self.cur_header = 0

    def __readVideo(self, start_frame, stride):
        """
        __readVideo : multithreading. read video, extract frame and store in self.frameQ
        """
        print('ROI_image_stream : Video IO Thread started\n')
        self._rewindPlayHeader()
        for i in range(start_frame):
            ret = self.vc.grab()
            self.cur_header += 1
        while True:
            if not self.frameQ.full():
                if self.cur_header >= self.num_frame:
                    break
                ret, frame = self.vc.read()
                self.cur_header += 1
                frame = cv.bitwise_and(frame, frame, mask=self.global_mask)
                self.frameQ.put((self.cur_header-1, frame))
                # skip other frames
                for i in range(stride-1):
                    ret = self.vc.grab()
                    self.cur_header += 1
            else:
                time.sleep(0.1)
        print('ROI_image_stream : Video IO Thread stopped\n')

    def __processFrames(self):
        """
        __processFrames : multithreading. extract ROI from frame stored in self.frameQ and store in self.blobQ
        """
        print('ROI_image_stream : ROI extraction Thread started\n')
        # run until frameQ is empty and thread is dead 
        while not(self.frameQ.empty()) or self.vcIOthread.is_alive():
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
            image = cv.absdiff(image, self.background)
        else:
            image = cv.addWeighted(
                cv.absdiff(image, self.background), weight_mediandiff,
                cv.absdiff(image, self.pastFrameImage), weight_recentdiff,
                0)

        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        binaryImage = cv.threshold(image, self.foregroundModel['animalThreshold'], 255, cv.THRESH_BINARY)[1]

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
            norm.cdf(animalSize + self.foregroundModel['animalSize']['sd'] * 0.1, self.foregroundModel['animalSize']['median'], self.foregroundModel['animalSize']['sd'])
            - norm.cdf(animalSize - self.foregroundModel['animalSize']['sd'] * 0.1, self.foregroundModel['animalSize']['median'], self.foregroundModel['animalSize']['sd']),
            1e-10*np.ones(animalSize.shape)], axis=0)
        L_Convexity = np.max([
            norm.cdf(animalConvexity + self.foregroundModel['animalConvexity']['sd'] * 0.1, self.foregroundModel['animalConvexity']['median'],self.foregroundModel['animalConvexity']['sd'])
            - norm.cdf(animalConvexity - self.foregroundModel['animalConvexity']['sd'] * 0.1, self.foregroundModel['animalConvexity']['median'],self.foregroundModel['animalConvexity']['sd']),
            1e-10*np.ones(animalConvexity.shape)], axis=0)
        L_Circularity = np.max([
            norm.cdf(animalCircularity + self.foregroundModel['animalCircularity']['sd'] * 0.1, self.foregroundModel['animalCircularity']['median'],self.foregroundModel['animalCircularity']['sd'])
            - norm.cdf(animalCircularity - self.foregroundModel['animalCircularity']['sd'] * 0.1, self.foregroundModel['animalCircularity']['median'],self.foregroundModel['animalCircularity']['sd']),
            1e-10*np.ones(animalCircularity.shape)], axis=0)

        likelihoods = np.log(L_Size) + np.log(L_Convexity) + np.log(L_Circularity)

        # If no blob is found, then return None as output
        if len(likelihoods) == 0:
            return None

        # If prevPoint is provided, then use previous location to calculate more accurate loglikelihood
        if prevPoint is not None:
            distance = np.sum((prevPoint - centers)**2, axis=1)**0.5
            L_Distance = np.max([
                norm.cdf(distance + self.foregroundModel['p2pDisplacement']['sd'] * 0.1, self.foregroundModel['p2pDisplacement']['median'],
                         self.foregroundModel['p2pDisplacement']['sd'])
                - norm.cdf(distance - self.foregroundModel['p2pDisplacement']['sd'] * 0.1, self.foregroundModel['p2pDisplacement']['median'],
                           self.foregroundModel['p2pDisplacement']['sd']),
                1e-10 * np.ones(distance.shape)], axis=0)  # 14.530208395578228, 13.609712279904377
            L_Distance = 0.1 * L_Distance # penalty term for fixing to a wrong location
            likelihoods += np.log(L_Distance)

        return centers[np.argmax(likelihoods)]

    def __denoiseBinaryImage(self, binaryImage):
        """
        __denoiseBinaryImage : remove small artifacts from the binary Image using the cv.morphologyEx function
        :param binaryImage: boolean image
        :return: denoised binary image
        """
        # opening -> delete noise : erode and dilate
        # closing -> make into big object : dilate and erode
        denoisedBinaryImage = binaryImage
        denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_OPEN, self.__getKernel(3))
        denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_OPEN, self.__getKernel(5))
        denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_CLOSE, self.__getKernel(10))
        # denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_OPEN, self.__getKernel(7))
        # denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_OPEN, self.__getKernel(12))
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

    def __getKernel(self, size):
        size = int(size)
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size),
                                        ((int((size - 1) / 2), int((size - 1) / 2))))

class ROI_image_stream_noblob():
    def __init__(self,path_data, ROI_size):
        """
        __init__ : initialize ROI image extraction stream object
        ****************************************************************
        path_data : Path object : path of the video or a folder containing the video
        ROI_size : width/height of the extracting ROI. This must be an even number.
        setMask : if True, ROI selection screen appears.
        """
        # Setup VideoCapture
        self.path_video = self.__readVideoPath(path_data)
        self.vc = cv.VideoCapture(str(self.path_video))
        self.frame_size = (int(self.vc.get(cv.CAP_PROP_FRAME_HEIGHT)), int(self.vc.get(cv.CAP_PROP_FRAME_WIDTH)))
        self.num_frame = int(self.vc.get(cv.CAP_PROP_FRAME_COUNT))
        self.cur_header = 0
        self.fps = int(self.vc.get(cv.CAP_PROP_FPS))

        # Multithreading
        self.isMultithreading = False

        # Extra data
        self.roiCoordinateData = []

        # ROI size
        self.ROI_size = ROI_size
        self.half_ROI_size = int(np.round(self.ROI_size/2))
        if ROI_size%2 != 0 :
            raise(BaseException('ROI_size is not dividable with 2!'))

        # Set Mask
        self.global_mask = 255 * np.ones(self.frame_size, dtype=np.uint8)

    def setRoiCoordinateData(self, data):
        """

        :param data: data
            n x 3 numpy data
            col0 : frame number
            col1 : row
            col2 : col
        :return:
        """
        self.roiCoordinateData = data

    def setGlobalMask(self, mask=[]):
        if not mask:
            mask_position = cv.selectROI('Select ROI', self.getFrame(0, applyGlobalMask=False))
            cv.destroyWindow('Select ROI')
            self.global_mask = np.zeros(self.frame_size, dtype=np.uint8)
            self.global_mask[mask_position[1]:mask_position[1] + mask_position[3],
            mask_position[0]:mask_position[0] + mask_position[2]] = 255
        else:
            self.global_mask = mask
    def getGlobalMask(self):
        return self.global_mask
    def startROIextractionThread(self, start_frame, stride=5):
        """
        startROIextractionThread : start ROI extraction Thread for continuous processing.
            When called, two thread (video read and opencv ROI detection) is initiated.
            Processed ROI is stored in self.blobQ
        --------------------------------------------------------------------------------
        stride : integer : The function read one frame from from every (stride) number of frames
        """
        # Video IO Thread and Queue
        self.frameQ = Queue(maxsize=200)
        self.vcIOthread = Thread(target=self.__readVideo, args=(start_frame, stride,))
        self.vcIOthread.daemon = True # indicate helper thread
        if not self.vcIOthread.is_alive():
            self.vcIOthread.start()

        self.isMultithreading = True
    def getROIImage(self):
        """
        getROIImage : run __findBlob(image)
        ----------------------------------------------------------------
        chosen_image : 3D numpy array : ROI frame
        blob_center : (r,c) : location of the center of the blob
        """

        if not self.isMultithreading: # if multithreading is not used
            raise(TypeError('getROIImage() missing 1 required positional argument: \'frame_number\'\n'
                            'If you are trying to use this function as multithreading, check if you called startROIextractionThread()'))


        frame_number, image = self.frameQ.get(timeout=10)


        data_index = np.where(self.roiCoordinateData[:,0] == frame_number)[0]
        if len(data_index) ==0 :
            raise(BlobDetectionFailureError('No Blob'))

        blob_center_row = self.roiCoordinateData[data_index[0], 1]
        blob_center_col = self.roiCoordinateData[data_index[0], 2]

        # Create expanded version of the original image.
        # In this way, we can prevent errors when center of the ROI is near the boarder of the image.
        expanded_image = cv.copyMakeBorder(image, self.half_ROI_size, self.half_ROI_size, self.half_ROI_size,
                                           self.half_ROI_size,
                                           cv.BORDER_CONSTANT, value=[0, 0, 0])

        chosen_image = expanded_image[
                       blob_center_row - self.half_ROI_size + self.half_ROI_size : blob_center_row + self.half_ROI_size + self.half_ROI_size,
                       blob_center_col - self.half_ROI_size + self.half_ROI_size : blob_center_col + self.half_ROI_size + self.half_ROI_size,:]

        return (chosen_image, [blob_center_row, blob_center_col])
    def getFrame(self, frame_number, applyGlobalMask = True):
        """
        getFrame : return original frame
        -------------------------------------------------------------------------------------
        frame_number : int : frame to process
        *Notice : this code, which look through the whole video from the beginning looks very
        inefficient, but it guarantee the retrieved frame is accurate. See opencv's Issue #9053
        -------------------------------------------------------------------------------------
        returns frame
        """
        # check if the input is int
        if type(frame_number) is not int:
            raise(BaseException('ROI_image_stream : frame_number must be an integer'))
        self._rewindPlayHeader()
        image = None
        while self.cur_header != frame_number+1:
            ret, image = self.vc.read()
            if not ret:
                raise(BaseException(f'ROI_image_stream : Can not retrieve frame # {frame_number}'))
            else:
                self.cur_header += 1
        if applyGlobalMask:
            image = cv.bitwise_and(image, image, mask=self.global_mask)
        return image
    def _rewindPlayHeader(self):
        """
        _rewindPlayHeader : rewind the play header of the VideoCapture object
        """
        self.vc.set(cv.CAP_PROP_POS_FRAMES, 0)
        if self.vc.get(cv.CAP_PROP_POS_FRAMES) != 0:
            raise(BaseException('ROI_image_stream : Can not set the play header to the beginning'))
        self.cur_header = 0
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
    def __readVideo(self, start_frame, stride):
        """
        __readVideo : multithreading. read video, extract frame and store in self.frameQ
        """
        print('ROI_image_stream : Video IO Thread started\n')
        self._rewindPlayHeader()
        for i in range(start_frame):
            ret = self.vc.grab()
            self.cur_header += 1
        while True:
            if not self.frameQ.full():
                if self.cur_header >= self.num_frame:
                    break
                ret, frame = self.vc.read()
                if not ret:
                    print('ROI_image_stream : wrong total frame number. early finishing __readVideo()\n')
                    self.num_frame = self.cur_header - stride# frame from cur_header does not contain frame. so cur_header - stride is the last valid frame
                    break
                self.cur_header += 1
                frame = cv.bitwise_and(frame, frame, mask=self.global_mask)
                self.frameQ.put((self.cur_header-1, frame))
                # skip other frames
                for i in range(stride-1):
                    ret = self.vc.grab()
                    self.cur_header += 1
            else:
                time.sleep(0.1)
        print('ROI_image_stream : Video IO Thread stopped\n')

class BlobDetectionFailureError(Exception):
    """Error class for blob detection"""
