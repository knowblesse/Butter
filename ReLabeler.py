"""
Read the video and buttered csv data, check, and relabel if necessary
"""
from pathlib import Path
import cv2 as cv
from ROI_image_stream import vector2degree
import numpy as np

# Constants
#TANK_PATH = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210622-180202_PL')
TANK_PATH = Path('D:/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210813-182242_IL')

# Find the path to the video 
if sorted(TANK_PATH.glob('*.mkv')): # path contains video.mkv
    # TODO: if there are multiple video files, raise the error
    path_video = next(TANK_PATH.glob('*.mkv'))
    print('ReLabeler : found *.mkv')
elif sorted(TANK_PATH.glob('*.avi')): # path contains video.avi
    path_video = next(TANK_PATH.glob('*.avi'))
    print('ReLabeler : found *.avi')
elif sorted(TANK_PATH.glob('*.mp4')): # path contains video.avi
    path_video = next(TANK_PATH.glob('*.mp4'))
    print('ReLabeler : found *.mp4')
else:
    raise(BaseException(f'ReLabeler : Can not find video file in {TANK_PATH}'))

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
    cv.putText(image, f'{current_frame} - {label_index/data.shape[0]*100:.2f}% - Excursion {np.sum(possibleExcursion)}', [0,int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)-1)],fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=[255,255,255], thickness=1)
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

while key!=ord('q'):
    cv.imshow('Main', labelObject.image)
    key = cv.waitKey(1)
    if key == ord('a'): # backward 0 min
        current_label_index = int(np.max([0, current_label_index - (60*lps)]))
        refreshScreen()
    elif key == ord('f'): # forward 1 min
        current_label_index = int(np.min([data.shape[0]-1, current_label_index + (60*lps)]))
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
        else:
            print('ReLabeler : No More Error Frame!')
    elif key == ord('w'): # read the next possible excursion
        foundExcursionIndex = np.argmax(np.abs(velocity))
        current_label_index = foundExcursionIndex
        refreshScreen()

    if labelObject.isLabeled:
        data[current_label_index,1] = labelObject.start_coordinate[1]
        data[current_label_index,2] = labelObject.start_coordinate[0]
        data[current_label_index,3] = vector2degree(
                labelObject.start_coordinate[1],
                labelObject.start_coordinate[0],
                labelObject.end_coordinate[1],
                labelObject.end_coordinate[0])
        labelObject.initialize(getFrame(current_label_index))

cv.destroyWindow('Main')
np.savetxt(str(path_csv), data,fmt='%d',delimiter='\t')
