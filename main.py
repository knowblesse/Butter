import datetime
import time
from pathlib import Path
from VideoProcessor import VideoProcessor
import numpy as np

# Path for the folder which has the video
video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21AUG3/21AUG3-211026-171617')

# Path for the model folder 
model_path = Path('/home/knowblesse/VCF/butter/Models/butterNet_V2')

# Create VideoProcessor Instance
vp = VideoProcessor(video_path, model_path)

# Build a foreground Model of the animal
vp.buildForegroundModel()

# Check the starting frame. (if the animal is present from the beginning of the video, just type zero
vp.checkStartPosition()

# Start the processing
starttime = time.time()
vp.run()
print(str(datetime.timedelta(seconds=time.time() - starttime)))
vp.save()
print('Saved!\n\n')
