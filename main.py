import datetime
import time
from pathlib import Path
from VideoProcessor import VideoProcessor
from tkinter.filedialog import askdirectory

# Path for the folder which has the video
video_path = Path(askdirectory())

# Path for the model folder 
model_path = Path('./Model/Model_230201_1649')

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
