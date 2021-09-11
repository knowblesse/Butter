import datetime
import time
from pathlib import Path
from VideoProcessor import VideoProcessor


video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210622-180202')
model_path = Path('./Model_210828_2000epoch')


vp = VideoProcessor(video_path, model_path)
vp.checkStartPosition()

starttime = time.time()
vp.run()
print(str(datetime.timedelta(seconds=time.time() - starttime)))
