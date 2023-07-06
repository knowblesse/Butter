import datetime
import time
from pathlib import Path
from Butter import Butter
from tkinter.filedialog import askdirectory

# Path for the folder which has the video
video_path = Path(r"C:\VCF\butter\debug\Lobster_Recording-200319-161008_20JUN1-200814-120239_Vid1.avi")

# Path for the model folder
model_path = Path('./Models/Model_230201_1649')

# Create VideoProcessor Instance
butter = Butter(video_path, model_path)

butter.setGlobalMask()

## Save BG FG

# butter.saveSampleFrames()
#
# butter.buildBackgroundModel()
#
# butter.buildForegroundModel()
#
# import pickle
# with open('fg.pk', 'wb') as f:
#     pickle.dump(butter.getForegroundModel(), f)
# with open('bg.pk', 'wb') as f:
#     pickle.dump(butter.getBackgroundModel(draw=False), f)

# Load BG FG
import pickle
with open('fg.pk', 'rb') as f:
    butter.setForegroundModel(pickle.load(f))
with open('bg.pk', 'rb') as f:
    butter.setBackgroundModel(pickle.load(f))

# Start Position
butter.checkStartPosition()

# Start the processing
starttime = time.time()
butter.run()
print(str(datetime.timedelta(seconds=time.time() - starttime)))
butter.save()
print('Saved!\n\n')
