"""
Read the video and buttered csv data, check, and relabel if necessary
"""
from pathlib import Path
import cv2 as cv
from butterUtil import vector2degree
import numpy as np
from NetworkTraining.checkPreviousDataset import checkPreviousDatddaset

# Constants
TANK_PATH = Path(r'D:\Data\Lobster\Lobster_Recording-200319-161008\20JUN1\#20JUN1-200901-105519_PL')
base_network = 'mobilenet_v2'

if base_network == 'mobilenet_v2':
    base_network_inputsize = 224
elif base_network == 'inception_v3':
    base_network_inputsize = 300
else:
    raise(BaseException('Not implemented'))

# Find the path to the video
vidlist = []
vidlist.extend([i for i in TANK_PATH.glob('*.mkv')])
vidlist.extend([i for i in TANK_PATH.glob('*.avi')])
vidlist.extend([i for i in TANK_PATH.glob('*.mp4')])
vidlist.extend([i for i in TANK_PATH.glob('*.mpg')])
if len(vidlist) == 0:
    raise(BaseException(f'ReLabeler : Can not find video in {TANK_PATH}'))
elif len(vidlist) > 1:
    raise(BaseException(f'ReLabeler : Multiple video files found in {TANK_PATH}'))
else:
    path_video = vidlist[0]

# Find the csv to the video
if sorted(TANK_PATH.glob(str(path_video.stem)+'_buttered.csv')):
    path_csv = next(TANK_PATH.glob(str(path_video.stem)+'_buttered.csv'))
else:
    raise (BaseException(f'ReLabeler : Can not find buttered file in {TANK_PATH}'))

# Load the video and the label data
vid = cv.VideoCapture(str(path_video))
data = np.loadtxt(str(path_csv))
num_frame = vid.get(cv.CAP_PROP_FRAME_COUNT)
fps = vid.get(cv.CAP_PROP_FPS)
lps = fps/data[1,0] # labels per second
current_label_index = 0

# Load the Dataset data to append new data directly.
datasetLocation = Path('./').absolute().parent/'NetworkTraining/Dataset'
(dataset_csv, dataset_number) = checkPreviousDataset(datasetLocation = datasetLocation)

# Find the excursion
velocity = ((data[1:,1] - data[0:-1,1]) ** 2 + (data[1:,2] - data[0:-1,2]) ** 2) ** 0.5
velocity = np.append(velocity, 0)
possibleExcursion = np.abs(velocity) > (np.mean(velocity) + 3*np.std(velocity))

# Main UI functions and callbacks
def getFrame(label_index):
    current_frame = int(data[label_index,0])
    vid.set(cv.CAP_PROP_POS_FRAMES, current_frame)
    ret, image = vid.read()
    if not ret:
        raise(BaseException('Can not read the frame'))
    cv.putText(image, f'{current_frame} - {label_index/data.shape[0]*100:.2f}% - Excursion {np.sum(possibleExcursion)}', [0,int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)-10)],fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=[255,255,255], thickness=1)
    cv.putText(image, 'a/f : +-1 min | s/d : +-1 label | e : error frame | w : excursion | q : quit', [0,15], fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=[255,255,255], thickness=1)
    if data[label_index,1] != -1:
        cv.circle(image, (round(data[label_index,2]), round(data[label_index,1])), 3, [0,0,255], -1 )
        cv.line(image, (round(data[label_index,2]), round(data[label_index,1])), (round(data[label_index,2] + 30*np.cos(np.deg2rad(data[label_index,3]))), round(data[label_index,1] + 30*np.sin(np.deg2rad(data[label_index,3])))), [0,255,255], 2)
    return image

class LabelObject:
    def __init__(self):
        self.isLabeled = True # default is True. if r button is pressed, initialized to False.
        self.active = False
    def initialize(self, image):
        self.image_org = image
        self.image = image
        self.start_coordinate = []
        self.end_coordinate = []
        self.active = False
        self.isLabeled = False

def drawLine(event, x, y, f, obj):
    if not obj.isLabeled: # if not labeled, react to the mouse event
        if event == cv.EVENT_LBUTTONDOWN:
            obj.start_coordinate = [x, y]
            obj.active = True
        elif event == cv.EVENT_LBUTTONUP:
            obj.active = False
            obj.isLabeled = True
            obj.end_coordinate = [x, y]
        elif event == cv.EVENT_MOUSEMOVE:
            if obj.active:
                obj.image = cv.line(obj.image_org.copy(), obj.start_coordinate, [x, y], [255,0,0], 2)

# Start Main UI 
key = ''
labelObject = LabelObject()
cv.namedWindow('Main')
cv.setMouseCallback('Main', drawLine, labelObject)
labelObject.initialize(getFrame(current_label_index))

def refreshScreen():
    labelObject.initialize(getFrame(current_label_index))
    possibleExcursion[current_label_index] = False
    velocity[current_label_index] = 0

def saveROIData(dataset_csv):
    # Get Current Frame
    current_frame_number = int(data[current_label_index, 0])
    vid.set(cv.CAP_PROP_POS_FRAMES, current_frame_number)
    ret, image = vid.read()
    if not ret:
        raise (BaseException('Can not read the frame'))

    # Get some variables to cut the frame into ROI size
    ROI_size = base_network_inputsize
    r = round(data[current_label_index, 1])
    c = round(data[current_label_index, 2])
    theta = data[current_label_index, 3]

    # Calculate center offset
    r_offset = np.random.randint(-1 / 4 * ROI_size, 1 / 4 * ROI_size)
    c_offset = np.random.randint(-1 / 4 * ROI_size, 1 / 4 * ROI_size)

    # Calculate crop range
    r_range = [r - int(ROI_size / 2) + r_offset, r + int(ROI_size / 2) + r_offset]
    c_range = [c - int(ROI_size / 2) + c_offset, c + int(ROI_size / 2) + c_offset]

    # Cut the image
    half_ROI_size = int(ROI_size/2)
    expanded_image = cv.copyMakeBorder(image, half_ROI_size, half_ROI_size, half_ROI_size, half_ROI_size,
                                       cv.BORDER_CONSTANT, value=[0, 0, 0])
    ROI_image = expanded_image[r_range[0]+half_ROI_size:r_range[1]+half_ROI_size, c_range[0]+half_ROI_size:c_range[1]+half_ROI_size, :]

    cv.imwrite(str(datasetLocation/Path(f'Dataset_{dataset_csv.shape[0]:04d}.png')), ROI_image)

    new_data_in_roi = np.expand_dims((int(ROI_size / 2) - r_offset, int(ROI_size / 2) - c_offset, theta), 0)

    dataset_csv = np.vstack((dataset_csv[:, 1::], new_data_in_roi))
    dataset_csv = np.hstack((np.expand_dims(np.arange(dataset_csv.shape[0]), 1), dataset_csv))
    np.savetxt(datasetLocation/Path('Dataset.csv'), dataset_csv, delimiter=',')
    print(f'ReLabeler : New data is appended. Now total {dataset_csv.shape[0]:d} data is in the dataset')
    return dataset_csv

while key!=ord('q'):
    cv.imshow('Main', labelObject.image)
    key = cv.waitKey(1)
    if key == ord('a'): # backward 0 min
        current_label_index = int(np.max([0, current_label_index - (10*lps)]))
        refreshScreen()
    elif key == ord('f'): # forward 1 min
        current_label_index = int(np.min([data.shape[0]-1, current_label_index + (10*lps)]))
        refreshScreen()
    elif key == ord('s'): # backward 1 label
        current_label_index = int(np.max([0, current_label_index - 1]) )
        refreshScreen()
    elif key == ord('d'): # forward 1 label
        current_label_index = int(np.min([data.shape[0]-1, current_label_index + 1]))
        refreshScreen()
    elif key == ord('e'): # read the next error
        foundErrorIndex = np.where(data[:,1] == -1)[0]
        if len(foundErrorIndex) > 0:
            current_label_index = foundErrorIndex[0]
            refreshScreen()
            print(f'Relabeler : {foundErrorIndex.shape[0]} Error Frame(s) Left')
        else:
            print('ReLabeler : No More Error Frame!')
    elif key == ord('w'): # read the next possible excursion
        foundExcursionIndex = np.argmax(np.abs(velocity))
        current_label_index = foundExcursionIndex
        refreshScreen()
    elif key == ord('g'):
        dataset_csv = saveROIData(dataset_csv)
        print('Relabeler : Saved!')
    if labelObject.isLabeled:
        if not (labelObject.start_coordinate == labelObject.end_coordinate):
            try:
                data[current_label_index,1] = labelObject.start_coordinate[1]
                data[current_label_index,2] = labelObject.start_coordinate[0]
                data[current_label_index, 3] = vector2degree(
                    labelObject.start_coordinate[1],
                    labelObject.start_coordinate[0],
                    labelObject.end_coordinate[1],
                    labelObject.end_coordinate[0])
                labelObject.initialize(getFrame(current_label_index))
            except IndexError :
                labelObject.isLabeled = False
        else: # did not drag (can not retrieve degree)
            refreshScreen()

cv.destroyWindow('Main')
np.savetxt(str(path_csv), data,fmt='%d',delimiter='\t')
