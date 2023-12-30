import datetime
import time
import pickle
from pathlib import Path
from Butter import Butter
from tkinter.filedialog import askdirectory
import numpy as np

import requests

for base_num, base in enumerate([Path(r'D:\Data_fib\Robot Predator\Rtest5')]):

    for num, i in enumerate(base.glob('R*')):
        # Path for the folder which has the video
        video_path = i

        # Path for the model folder
        model_path = Path('./Models/butterNet_V2.3')

        # Load tracking Data
        trackingData = np.loadtxt(next(video_path.glob('tracking.csv')), delimiter=',', dtype=np.int32)
        roiCooridnate = trackingData[:, [0,4,3]]

        # Create VideoProcessor Instance
        butter = Butter(video_path, model_path, process_fps=12, roiCoordinateData=roiCooridnate)

        # Set global mask
        #butter.setGlobalMask()

        # Start the processing
        starttime = time.time()
        butter.run()
        print(str(datetime.timedelta(seconds=time.time() - starttime)))
        butter.save()
        print('Saved!\n\n')

requests.get(
        r"https://api.telegram.org/bot5269105245:AAE9AnATgUo2swh4Tyr4Fk7wdSVz3SqBS_4/sendMessage?chat_id=5520161508&text=DONE")