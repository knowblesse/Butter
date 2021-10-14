
import datetime
import time
from pathlib import Path
from VideoProcessor import VideoProcessor
import numpy as np

video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210813-182242_IL')

######video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210803-182450')
# brightness is changed after 4min..

####### video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN2/#21JAN2-210419-175714')
# need to be corrected. Rat sometimes vanish

# video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN2/#21JAN2-210423-194013')
# 558 no ROI

# video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN2/#21JAN2-210428-195618')
# 453 no ROI

# video_path = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN2/#21JAN2-210503-180009')
# 266

model_path = Path('./Model_210911_6000epoch')
vp = VideoProcessor(video_path, model_path)
vp.trainBackgroundSubtractor()
vp.checkStartPosition()

starttime = time.time()
vp.run()
print(str(datetime.timedelta(seconds=time.time() - starttime)))
vp.save()
print('Saved!\n\n')

vp.checkResult(np.arange(1000,10000,200))
