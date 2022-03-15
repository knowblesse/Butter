import datetime
import time
from pathlib import Path
from VideoProcessor import VideoProcessor
import numpy as np


video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/20JUN1/#20JUN1-200831-110125_PL')

model_path = Path('/home/knowblesse/VCF/butter/Model_210911_7500epoch')
vp = VideoProcessor(video_path, model_path)
vp.buildForegroundModel()
vp.checkStartPosition()

starttime = time.time()
vp.run()
print(str(datetime.timedelta(seconds=time.time() - starttime)))
vp.save()
print('Saved!\n\n')
