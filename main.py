import datetime
import time
from pathlib import Path
from VideoProcessor import VideoProcessor
from tkinter.filedialog import askdirectory

# Path for the folder which has the video
video_path = Path(askdirectory())

# Path for the model folder 
model_path = Path('./Models/Model_230201_1649')

# Create VideoProcessor Instance
vp = VideoProcessor(video_path, model_path)

# Build a foreground Model of the animal
#vp.buildForegroundModel()

animalThreshold = 31
p2pDisplacement={'median': 17.20, 'sd': 312.87}
animalSize={'median': 2750, 'sd': 312.87}
animalConvexity={'median': 0.88, 'sd': 0.09}
animalCircularity={'median': 0.49, 'sd': 0.09}
vp.setForegroundModel(animalThreshold, p2pDisplacement, animalSize, animalConvexity, animalCircularity)


# Check the starting frame. (if the animal is present from the beginning of the video, just type zero
vp.checkStartPosition()

# Start the processing
starttime = time.time()
vp.run()
print(str(datetime.timedelta(seconds=time.time() - starttime)))
vp.save()
print('Saved!\n\n')
