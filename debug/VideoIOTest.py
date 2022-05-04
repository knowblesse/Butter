import ffmpeg
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from ROI_image_stream import *

from time import time

videoPath = Path('/home/knowblesse/Desktop/22-03-28(월)_04.mpg')
__funcName__ = 'test_ROI_extraction'

videoObject = ffmpeg.input(str(videoPath))

probe = ffmpeg.probe(str(videoPath))
info = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']

if len(info) == 0:
    raise(BaseException(__funcName__ +  'No video stream found'))
elif len(info) > 1:
    raise(BaseException(__funcName__ +  'Multiple video streams found'))
info = info[0]

height = int(info['height'])
width = int(info['width'])
duration = float(info['duration']) # length of the stream in second = duration_ts * time_base
duration_ts = int(info['duration_ts']) # number of the frame
time_base = info['time_base']

r_frame_rate = info['r_frame_rate'] # the Least Common Multiple(LCM) of all framerates in the stream
avg_frame_rate = info['avg_frame_rate'] # Num Frame / Duration (?)

if r_frame_rate != avg_frame_rate:
    raise(BaseException(__funcName__ + 'Can not obtain framerate. Probably the video has the variable frame rate?'))
avg_frame_rate = eval(avg_frame_rate)
print(f'num frame : {duration*avg_frame_rate}')
# 55340
# Option 1 : OpenCV calling all frames
cap = cv.VideoCapture(str(videoPath))
startTime = time()
EOF = False
fr = 0
while not EOF:
    fr += 1
    ret, frame = cap.read()
    if not ret:
        EOF = True
print(f'Option 1 Time {time() - startTime}')

# Option 2 : ffmpeg methods

import ffmpeg

videoObject = ffmpeg.input(str(videoPath))

probe = ffmpeg.probe(str(videoPath))
info = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
info = info[0]

height = int(info['height'])
width = int(info['width'])

## Option 2-1 : ffmpeg : Directly reading bgr image from yuv420p video
# Conversion is done inside the ffmpeg side
out = (
    ffmpeg
    .input(str(videoPath))
    .output('pipe:', format='rawvideo', pix_fmt = 'bgr24')
    .global_args('-loglevel', 'error', '-hide_banner')
    .run_async(pipe_stdout=True)
)

startTime = time()
EOF = False
fr = 0
while not EOF:
    fr += 1
    frame = out.stdout.read(int(height * width * 3))
    if not len(frame):
        EOF = True
    else:
        img = np.frombuffer(frame, np.uint8).reshape([height, width, 3])

print(f'Option 2-1 Time {time() - startTime}')

# Option 2-2 : ffmpeg : Reading yuv420p video and converting from the opencv side
# Credit to roninpawn
out = (
    ffmpeg
    .input(str(videoPath))
    .output('pipe:', format='rawvideo', pix_fmt = 'yuv420p')
    .global_args('-loglevel', 'error', '-hide_banner')
    .run_async(pipe_stdout=True)
)

startTime = time()
EOF = False
fr = 0
while not EOF:
    fr += 1
    frame = out.stdout.read(int(height * width * 1.5))
    if not len(frame):
        EOF = True
    else:
        img = np.frombuffer(frame, np.uint8).reshape([height*3//2, width])
        img = cv.cvtColor(img, cv.COLOR_YUV2BGR_I420)

print(f'Option 2-2 Time {time() - startTime}')