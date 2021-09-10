"""
GenerateTrainingDataset.py
@Knowblesse 2021
21 AUG 25
Generate image dataset for network training.
- dataset is imported from Matlab-based HeadDirectionLabeler
- a video file and exported .csv is used to generate dataset
- Further Network training algorithm uses image produced from this script and csv file.
- I chose image (not video) input for network training, because this way I can manually reject wrong datasets
- Data augmentation is not implemented in this script.
- Inputs
    -new_data_folder : path to the labeled video and csv file
- Outputs
    - Numbered Image
    - one csv file containing row, col, head angle
"""

import numpy as np
import cv2 as cv
from pathlib import Path
from ROI_image_stream import ROI_image_stream
from checkPreviousDataset import checkPreviousDataset

################################################################
# Constants
################################################################
new_data_folder = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210622-180202')#Path.home() / 'VCF/butter/dataset'
base_network = 'mobilenet'
istream_threshold = 70
################################################################
# Setup
################################################################
if base_network == 'mobilenet':
    base_network_inputsize = 224
elif base_network == 'inception_v3':
    base_network_inputsize = 300
else:
    raise(BaseException('Not implemented'))

################################################################
# Check Previous Dataset
################################################################
(dataset_csv, dataset_number) = checkPreviousDataset()

################################################################
# Load csv file from new dataset
################################################################
try:
    csv_data = np.loadtxt(str(next(new_data_folder.glob('*.csv'))),delimiter=',')
    if csv_data.shape[1] != 4:
        raise(BaseException('Label csv file has wrong number of column'))
    labeledIndex = np.where(csv_data[:,3] == 1)[0]
    new_data_csv = csv_data[labeledIndex, :-1]
except:
    raise(BaseException('Labeled csv file reading error'))

################################################################
# Process : 1. generate ROIed image data from video
#           2. transform labeled location into in-ROI location
################################################################
istream = ROI_image_stream(new_data_folder, ROI_size=base_network_inputsize)
istream.threshold = istream_threshold
istream.trainBackgroundSubtractor(stride=500)

new_data_in_roi = np.zeros((len(labeledIndex), 3))
for i, frame_number in enumerate(labeledIndex):
    chosen_image, coor = istream.getROIImage(frame_number)
    cv.imwrite(str(Path(f'./Dataset/Dataset_{dataset_number + i:04d}.png')),chosen_image)
    new_data_in_roi[i,0:2] = new_data_csv[i,0:2] - coor + base_network_inputsize / 2
    new_data_in_roi[i,2] = new_data_csv[i,2]

if 'y' == input(f'GenerateTrainingDataset : {new_data_in_roi.shape[0]:d} data will be added. Continue?(y/n)'):
    dataset_csv = np.vstack((dataset_csv[:,1::], new_data_in_roi))
    dataset_csv = np.hstack((np.expand_dims(np.arange(dataset_csv.shape[0]),1), dataset_csv))
    np.savetxt(Path('./Dataset/Dataset.csv'),dataset_csv, delimiter=',')
    print(f'GenerateTrainingDataset : {new_data_in_roi.shape[0]:d} data is appended. Now total {dataset_csv.shape[0]:d} data is in the dataset')
else:
    print('GenerateTrainingDataset : Aborted')