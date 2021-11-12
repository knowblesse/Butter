"""
Read the video and buttered csv data, Label each frame using interpolation, and save it into a video
"""
from pathlib import Path
import cv2 as cv
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
# Constants
TANK_PATH = Path('D:/Data/Lobster/Lobster_Recording-200319-161008/20JUN1/#20JUN1-200814-120239_PL')

# Find the path to the video
vidlist = []
vidlist.extend([i for i in TANK_PATH.glob('*.mkv')])
vidlist.extend([i for i in TANK_PATH.glob('*.avi')])
vidlist.extend([i for i in TANK_PATH.glob('*.mp4')])
if len(vidlist) == 0:
    raise(BaseException(f'SaveLabeledVideo : Can not find video in {TANK_PATH}'))
elif len(vidlist) > 1:
    raise(BaseException(f'SaveLabeledVideo : Multiple video files found in {TANK_PATH}'))
else:
    path_video = vidlist[0]

# Find the csv to the video
if sorted(TANK_PATH.glob(str(path_video.stem)+'_buttered.csv')):
    path_csv = next(TANK_PATH.glob(str(path_video.stem)+'_buttered.csv'))
else:
    raise (BaseException(f'SaveLabeledVideo : Can not find buttered file in {TANK_PATH}'))

# Load the video and meta data
vid = cv.VideoCapture(str(path_video))

# get total number of frame
#   I can not trust vid.get(cv.CAP_PROP_FRAME_COUNT), because sometime I can't retrieve the last frame with vid.read()
num_frame = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
vid.set(cv.CAP_PROP_POS_FRAMES, num_frame)
ret, _ = vid.read()
while not ret:
    print(f'SaveLabeledVideo : Can not read the frame from the last position. Decreasing the total frame count')
    num_frame -= 1
    vid.set(cv.CAP_PROP_POS_FRAMES, num_frame)
    ret, _ = vid.read()
fps = vid.get(cv.CAP_PROP_FPS)

# Load the label data
data = np.loadtxt(str(path_csv))
lps = fps/data[1,0] # labels per second

# Generate offset Head Degree Data
#   The head degree of an animal is stored in a range between 0 to 360 degree.
#   If I use the raw head direction data from the labeled dataset,
#   normal interpolation algorithm wouldn't correctly detect the change between near 0 or 360 degrees.
#   For example, changed value from 10 to 350 degrees might be only 20 degrees, but the interpolation
#   algorithm would say the value between these two points are 180 degrees, not 0 degree.
#   To compensate this, we need to add degree_offset_value to the origianl degree data.
#   Since degree values used to draw lines does not be affected when I add or subtract 360 degrees,
#   these offset won't change the result.
prev_head_direction = data[0,3]
degree_offset_value = np.zeros(data.shape[0])
for i in np.arange(1, data.shape[0]):
    if np.abs(data[i,3] - prev_head_direction) > 180:
        print(f'Degree Change : {prev_head_direction} --> {data[i,3]}')
        if data[i,3] > prev_head_direction:
            degree_offset_value[i:] -= 360
        else:
            degree_offset_value[i:] += 360
    prev_head_direction = data[i,3]

# Generate Interpolated Label Data
intp_x = interp1d(data[:,0], data[:,1], kind='linear')
intp_y = interp1d(data[:,0], data[:,2], kind='linear')
intp_d = interp1d(data[:,0], np.convolve(data[:,3]+ degree_offset_value, np.ones(5), 'same') / 5, kind='linear')

data_intp = np.stack([np.array(np.arange(num_frame)),
    intp_x(np.array(np.arange(num_frame))),
    intp_y(np.array(np.arange(num_frame))),
    intp_d(np.array(np.arange(num_frame)))],axis=1)

# Save Video
vid_out = cv.VideoWriter(
    str(path_video.parent / (path_video.stem + '_labeled.' + path_video.suffix)),
    cv.VideoWriter_fourcc(*'DIVX'),
    fps,
    (int(vid.get(cv.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))),
    isColor=True)

for i in tqdm(np.arange(num_frame)):
    i = int(i)
    vid.set(cv.CAP_PROP_POS_FRAMES, i)
    ret, image = vid.read()
    if not ret:
        raise(BaseException('Can not read the frame'))
    cv.circle(image, (round(data_intp[i,2]), round(data_intp[i,1])), 3, [0,0,255], -1 )
    cv.line(image, (round(data_intp[i,2]), round(data_intp[i,1])), (round(data_intp[i,2] + 30*np.cos(np.deg2rad(data_intp[i,3]))), round(data_intp[i,1] + 30*np.sin(np.deg2rad(data_intp[i,3])))), [0,255,255], 2)
    vid_out.write(image)

vid_out.release()