import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class ROI_image_stream():
    def __init__(self,path_dataset, ROI_size):
        # Parameter
        self.path_dataset = path_dataset
        self.path_video = self.path_dataset.glob('*.mkv')

        # setup VideoCapture
        self.vid = cv.VideoCapture(str(next(self.path_video)))
        
        # backgroundSubtractor
        self.isBackgroundSubtractorTrained = False

        # threshold
        self.threshold = 50

        # ROI size
        self.ROI_size = ROI_size
        self.half_ROI_size = int(np.round(self.ROI_size/2))
        if ROI_size%2 != 0 :
            raise(BaseException('ROI_size is not dividable with 2!'))

    def trainBackgroundSubtractor(self,stride=2000):
        """
        trainBackgroundSubtractor : train BackgroundSubtractorKNN
        ---------------------------------------------------------------- 
        stride : int : Frame stride for training purpose 
        """
        self.backSub = cv.createBackgroundSubtractorKNN()
        self.backSub.setShadowThreshold(0.01)
        self.backSub.setShadowValue(0)
        count = 0
        #TRAIN
        while True:
            self.vid.set(cv.CAP_PROP_POS_FRAMES, count)
            ret, image = self.vid.read()
            if image is None:
                print("End of the File")
                break
            else:
                masked_image = self.backSub.apply(cv.threshold(image, self.threshold, 0, cv.THRESH_TOZERO)[1])
                print(str(count) + 'th image is used for the training')
                count += stride
                continue
        self.isBackgroundSubtractorTrained = True

    def extractROIImage(self, frame_number, erosion_size=3, dilate_size=3, drawPlot=False, prevPoint=None, distance_criterion = 50):
        """
        extractROIImage : return ROI frame from the video
        ----------------------------------------------------------------
        frame_number : int or int numpy array : frame to process
        erosion_size : int : cv.erosion shape size
        dilate_size : int : cv.dilate shape size
        drawPlot : bool : draw each frame in image processing steps
        prevPoint : cv.KeyPoint object : previous position data to detect position excursion
        distance_criterion : int : distance criterion to detect position excursion
        ---------------------------------------------------------------- 
        outputFrame : 3D numpy array : ROI_size cut frame  
        center : 2D numpy array : location of the center of the blob
        """

        # TODO: multiple frame processing by accepting frame_number array
        # TODO: multiprocessing
        if not self.isBackgroundSubtractorTrained:
            raise(BaseException('BackgroundSubtractor is not trained'))

        self.vid.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, image = self.vid.read()
        image = cv.threshold(image, self.threshold, 0, cv.THRESH_TOZERO)[1]
        masked_image = self.backSub.apply(image,learningRate = 1e-6) # according to the document, 0 should freeze the backgroundsubtractor, but it fails. So we put very small number
        image = cv.copyMakeBorder(image, self.half_ROI_size, self.half_ROI_size, self.half_ROI_size, self.half_ROI_size, cv.BORDER_CONSTANT, value=[0, 0, 0])

        # 1. Erode :
        erosion_shape = cv.MORPH_RECT
        element_erd = cv.getStructuringElement(erosion_shape, (2*erosion_size+1,2*erosion_size+1),(erosion_size, erosion_size))
        masked_image_erd = cv.erode(masked_image,element_erd)

        # 2. Dilate
        dilate_shape = cv.MORPH_RECT
        element_dlt = cv.getStructuringElement(dilate_shape, (2*dilate_size+1,2*dilate_size+1),(dilate_size, dilate_size))
        masked_image_erd_dlt = cv.dilate(masked_image_erd, element_dlt)

        # 3. SimpleBlobDetector
        parameter = cv.SimpleBlobDetector_Params()
        parameter.filterByArea = True
        parameter.minArea = 1000 # this value defines the minimum size of the blob
        parameter.maxArea = 10000 # this value defines the maximum size of the blob
        parameter.filterByColor = True # I found one blog article sayint that this function does not work well. But, still using it
        parameter.blobColor = 255
        parameter.minDistBetweenBlobs = 10
        parameter.filterByInertia = False
        parameter.filterByConvexity = False
        detector = cv.SimpleBlobDetector_create(parameter)
        detected_blob = detector.detect(masked_image_erd_dlt)

        def drawDebugPlot():
            fig = plt.figure(1)
            fig.clf()
            ax = fig.subplots(2, 2)
            ax[0,0].imshow(image)
            ax[0,0].set_title('original')
            ax[0,1].imshow(masked_image)
            ax[0,1].set_title('masked_image')
            ax[1,0].imshow(masked_image_erd)
            ax[1,0].set_title('erode')
            ax[1,1].imshow(masked_image_erd_dlt)
            ax[1,1].set_title('erode + dilate')

        # For debuging purpose, Draw all image processing steps
        if drawPlot:
            drawDebugPlot()
        
        # Further process detected blobs
        max_blob_index = 0
        if len(detected_blob) > 1:# if multiple blob is detected, select the largest one
            max_blob_size = 0
            for i, blob in enumerate(detected_blob):
                if max_blob_size < blob.size:
                    max_blob_size = blob.size
                    max_blob_index = i
            print('Multiple blobs : %d detected in frame %d' % (len(detected_blob), frame_number))

        elif len(detected_blob) == 0 :
            drawDebugPlot()
            raise(BaseException("%d frame : No blob detected!" % frame_number))
        blob_center_row, blob_center_col = int(np.round(detected_blob[max_blob_index].pt[1])) , int(np.round(detected_blob[max_blob_index].pt[0]))
        chosen_image = image[
                       blob_center_row - self.half_ROI_size + self.half_ROI_size : blob_center_row + self.half_ROI_size + self.half_ROI_size,
                       blob_center_col - self.half_ROI_size + self.half_ROI_size : blob_center_col + self.half_ROI_size + self.half_ROI_size,:]

        # Further check whether the final blob's location is not far away from the previous one.
        # ==> prevent excursion
        if prevPoint is not None:
            if ((prevPoint[0] - detected_blob[max_blob_index].pt[0])**2 + (prevPoint[1] - detected_blob[max_blob_index].pt[1])**2)**0.5 > distance_criterion:
                raise(BaseException("excursion might occured"))
        return(chosen_image, [blob_center_row, blob_center_col])

################################################################
# Check Dataset New Images
################################################################
def checkDataSet(X,y):
    for idx in np.arange(X.shape[0]):
        r = 20
        plt.clf()
        plt.imshow(X[idx,:,:,:])
        # draw real
        plt.scatter(y[idx,1], y[idx,0],c='g')
        plt.plot([y[idx,1], y[idx,1] + r*np.cos(np.deg2rad(y[idx,2]))], [y[idx,0], y[idx,0] - r*np.sin(np.deg2rad(y[idx,2]))], lineWidth=3, color = 'g')
        plt.title(str(idx))
        # print output
        #print("%05d : (%03d, %03d)@%03d : (%03d, %03d)@%03d" % (int(idx), int(y_test[idx,0]), int(y_test[idx,1]), int(y_test[idx,2]), int(y_pred[idx,0]), int(y_pred[idx,1]), int(y_pred[idx,2])))
        plt.pause(0.3)
