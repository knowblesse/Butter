"""
Read the video and buttered csv data, check, and relabel if necessary
"""
from pathlib import Path
import cv2 as cv
from ROI_image_stream import vector2degree

# Constants
TANK_PATH = Path('./')

# Find the path to the video 
if sorted(TANK_PATH.glob('*.mkv')): # path contains video.mkv
    # TODO: if there are multiple video files, raise the error
    path_video = next(TANK_PATH.glob('*.mkv'))
    print('ROI_image_stream : found *.mkv')
elif sorted(TANK_PATH.glob('*.avi')): # path contains video.avi
    path_video = next(TANK_PATH.glob('*.avi'))
    print('ROI_image_stream : found *.avi')
else:
    raise(BaseException(f'ROI_image_stream : Can not find video file in {TANK_PATH}'))

# Find the csv to the video
if sorted(TANK_PATH.glob(str(path_video.stem)+'_buttered.csv')):
    path_csv = next(TANK_PATH.glob(str(path_video.stem)+'_buttered.csv'))

# Load the video and the label data
vid = cv.VideoCapture(str(path_video))
data = np.loadtxt(str(path_csv))
num_frame = vid.get(cv.CAP_PROP_FRAME_COUNT)
fps = vid.get(cv.CAP_PROP_FPS)
lps = fps/data[1,0] # labels per second
current_label_index = 0

# Main UI functions and callbacks
def getFrame(label_index):
    current_frame = data[label_index,0]
    vid.set(cv.CAP_PROP_POS_FRAMES, current_frame)
    ret, image = vid.read()
    if not ret:
        raise(BaseException('Can not read the frame'))
    cv.putText(image, str(current_frame), [0,vid.get(cv.CAP_PROP_FRAME_HEIGHT)],fontFace=10, color=[0,0,0])
    if data[label_index,1] != -1:
        cv.circle(image, (round(data[label_index,2]), round(data[label_index,1])), 3, [0,0,255], -1 )
        cv.line(img, (round(data[label_index,2]), round(data[label_index,1])), (round(data[label_index,2] + 30*np.cos(np.deg2rad(data[label_index,3]))), round(data[label_index,1] + 30*np.sin(np.deg2rad(data[label_index,3])))), [0,255,255], 2)
        
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
        self.active = True 
        self.isLabeled = False

def drawLine(event, x, y, f, obj):
    if not obj.isLabeled: # if not labeled, react to the mouse event
        if event == cv.EVENT_LBUTTONDOWN:
            obj.start_coordinate = [x, y]
            obj.active = True
        elif event == cv.EVENT_LBUTTONUP:
            obj.active = False
            obj.isLabeled = True
        elif event == cv.EVENT_MOUSEMOVE:
            if obj.active:
                obj.image = cv.line(obj.image_org.copy(), obj.start_coordinate, [x,y], [255,0,0], 2)

# Start Main UI 
key = ''
labelObject = LabelObject()
cv.namedWindow('Main')
cv.setMouseCallback('Main', drawLine, labelObject)

while key=='q':
    image = getFrame(current_label_index)
    cv.imshow('Main', image)
    key = cv.waitKey()
    if key == 'a': # backward 1 min
        current_label_index = np.max([0, current_label_index - (60*lps)]) 
    elif key == 'f': # forward 1 min
        current_label_index = np.min([data[-1,0], current_label_index + (60*lps)]) 
    elif key == 's': # backward 1 label
        current_label_index = np.max([0, current_label_index - 1]) 
    elif key == 'd': # forward 1 label
        current_label_index = np.min([data[-1,0], current_label_index + 1])
    elif key == 'r': # relabel
        labelObject.initialize(image)
        while not labelObject.isLabeled:
            cv.imshow('Main', labelObject.image)
            cv.waitKey(1)
        data[current_label,1] = labelObject.start_coordinate[1]
        data[current_label,2] = labelObject.start_coordinate[0]
        data[current_label,3] = vector2degree(
                labelObject.start_coordinate[1],
                labelObject.start_coordinate[0],
                labelObject.end_coordinate[1],
                labelObject.end_coordinate[0])
    elif key == 'e': # read the next error
        foundErrorIndex = np.where(data[:,1] == -1)[0]
        if len(foundErrorIndex) > 0:
            current_label_index = foundErrorIndex[0] 
        else:
            print('ReLabeler : No More Error Frame!')

