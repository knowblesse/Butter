import datetime
import time
import pickle
from pathlib import Path
from Butter import Butter
from tkinter.filedialog import askdirectory
import numpy as np

import requests

video_path = Path(askdirectory())
# Path for the model folder
model_path = Path('./Models/butterNet_V2.3')

# Load tracking Data
trackingData = np.loadtxt(next(video_path.glob('tracking.csv')), delimiter=',', dtype=np.int32)
roiCooridnate = trackingData[:, [0, 4, 3]]

# Create VideoProcessor Instance
butter = Butter(video_path, model_path, process_fps=12, roiCoordinateData=roiCooridnate)

# Set global mask
# butter.setGlobalMask()

# Start the processing
starttime = time.time()
butter.run()
print(str(datetime.timedelta(seconds=time.time() - starttime)))
butter.save()
print('Saved!\n\n')