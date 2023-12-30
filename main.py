import datetime
import time
import pickle
from pathlib import Path
from Butter import Butter
from tkinter.filedialog import askdirectory

# Path for the folder which has the video
video_path = Path(askdirectory())

# Path for the model folder
model_path = Path('./Models/butterNet_V2.3')

# Create VideoProcessor Instance
butter = Butter(video_path, model_path)

# Set global mask
butter.setGlobalMask()

## If Background and foreground model exist, use it.
if Path('fg.pk').exists() and Path('bg.pk').exists():
    with open('fg.pk', 'rb') as f:
        butter.setForegroundModel(pickle.load(f))
    with open('bg.pk', 'rb') as f:
        butter.setBackgroundModel(pickle.load(f))
else: # Make background and foreground model
    butter.saveSampleFrames()

    butter.buildBackgroundModel()

    butter.buildForegroundModel()

    with open('fg.pk', 'wb') as f:
        pickle.dump(butter.getForegroundModel(), f)
    with open('bg.pk', 'wb') as f:
        pickle.dump(butter.getBackgroundModel(draw=False), f)

# Start Position
butter.checkStartPosition()

# Start the processing
starttime = time.time()
butter.run()
print(str(datetime.timedelta(seconds=time.time() - starttime)))
butter.save()
print('Saved!\n\n')
