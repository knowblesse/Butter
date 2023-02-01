"""
Read the video and buttered csv data, Label each frame using interpolation, and save it into a video
"""
from pathlib import Path
import cv2 as cv
import numpy as np
from tqdm import tqdm
from tkinter.filedialog import askdirectory
from butterUtil import interpolateButterData
# Constants
TANK_PATH = Path(askdirectory())

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

data_intp = interpolateButterData(data, num_frame)

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

    idx = np.where(i==data_intp[:,0])[0]
    if idx.shape[0] != 0: # skip last several frames (can not do extrapolation)
        idx = idx[0]
        cv.circle(image, (round(data_intp[idx,2]), round(data_intp[idx,1])), 3, [0,0,255], -1 )
        cv.line(image, (round(data_intp[idx,2]), round(data_intp[idx,1])), (round(data_intp[idx,2] + 30*np.cos(np.deg2rad(data_intp[idx,3]))), round(data_intp[idx,1] + 30*np.sin(np.deg2rad(data_intp[idx,3])))), [0,255,255], 2)
    vid_out.write(image)

vid_out.release()
