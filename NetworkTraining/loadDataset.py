import cv2 as cv
import numpy as np
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt

def loadDataset(base_network):
    ################################################################
    # Setup
    ################################################################
    if base_network == 'mobilenet_v2':
        base_network_inputsize = 224
    elif base_network == 'inception_v3':
        base_network_inputsize = 300
    else:
        raise(BaseException('Not implemented'))

    ################################################################
    # Input data : y
    ################################################################
    try:
        csv_data = np.loadtxt(str(Path('./Dataset/Dataset.csv')),delimiter=',')
        y_raw = csv_data[:,1::]
    except:
        raise(BaseException('Label csv file reading error'))
    dataSize = y_raw.shape[0]

    ################################################################
    # Input data Processing
    ################################################################
    X = np.zeros((4 * dataSize,base_network_inputsize,base_network_inputsize,3))
    y = np.zeros((4 * dataSize,4))

    dataset_image = [x for x in sorted(Path('./Dataset').glob('*.png'))]

    if dataSize != len(dataset_image):
        raise(BaseException('TrainNetwork : Dataset size mismatch'))

    # Data augmentation
    for i, clip in enumerate(dataset_image):
        chosen_image = cv.imread(str(clip))
        X[i * 4 + 0, :, :, :] = chosen_image
        X[i * 4 + 1, :, :, :] = cv.flip(chosen_image, 0)# updown (row)
        X[i * 4 + 2, :, :, :] = cv.flip(chosen_image, 1)# leftright (col)
        X[i * 4 + 3, :, :, :] = cv.flip(chosen_image, -1)# both

        corr = y_raw[i, 0:2]
        y[i * 4 + 0, 0:2] = corr
        y[i * 4 + 1, 0:2] = [base_network_inputsize - corr[0], corr[1]] #updown (row)
        y[i * 4 + 2, 0:2] = [corr[0], base_network_inputsize - corr[1]] #leftright (col)
        y[i * 4 + 3, 0:2] = [base_network_inputsize - corr[0], base_network_inputsize - corr[1]]

        # Degree coding into 30 pixel away point
        r = 30
        y[i * 4 + 0, 2:4] = [
                y[i * 4 + 0, 0] + r*np.sin(np.deg2rad(y_raw[i,2])),
                y[i * 4 + 0, 1] + r*np.cos(np.deg2rad(y_raw[i,2]))]
        y[i * 4 + 1, 2:4] = [
                y[i * 4 + 1, 0] - r*np.sin(np.deg2rad(y_raw[i,2])),
                y[i * 4 + 1, 1] + r*np.cos(np.deg2rad(y_raw[i,2]))]
        y[i * 4 + 2, 2:4] = [
                y[i * 4 + 2, 0] + r*np.sin(np.deg2rad(y_raw[i,2])),
                y[i * 4 + 2, 1] - r*np.cos(np.deg2rad(y_raw[i,2]))]
        y[i * 4 + 3, 2:4] = [
                y[i * 4 + 3, 0] - r*np.sin(np.deg2rad(y_raw[i,2])),
                y[i * 4 + 3, 1] - r*np.cos(np.deg2rad(y_raw[i,2]))]

    #################################################################
    # Check loaded Dataset
    #################################################################

    # idx = np.random.randint(dataSize*4)

    # plt.clf()
    # plt.imshow(X[idx,:,:,:]/255)
    # plt.scatter(y[idx,1], y[idx,0])

    # r = 30
    # plt.plot([y[idx,1], y[idx,3]], [y[idx,0], y[idx,2]], LineWidth=3, color = 'r')

    #################################################################
    # Convert Dataset
    #################################################################
    if base_network == 'mobilenet_v2':
        X_conv = keras.applications.mobilenet_v2.preprocess_input(X)
    elif base_network == 'inception_v3':
        X_conv = keras.applications.inception_v3.preprocess_input(X)

    return (X_conv, y)
