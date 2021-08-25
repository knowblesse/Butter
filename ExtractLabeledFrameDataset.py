"""
ExtractLabeledFrameDataset.py
load video and csv file and load and save dataset
@2021 Knowblesse
"""

import multiprocessing
from pathlib import Path
import cv2 as cv
import numpy as np
import datetime

# project configuration
project_name = 'ImplantedRat'
path_video = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210629-183643/Lobster_Recording-210330-101307_21JAN5-210629-183643_Vid1.avi')
path_label = Path('/mnt/Data/Data/Lobster/Lobster_Recording-200319-161008/21JAN5/#21JAN5-210629-183643/Lobster_Recording-210330-101307_21JAN5-210629-183643_Vid1.avi_data.csv')
path_export = Path.joinpath(Path.home(),'VCF/butter/dataset/ImplantedRat')

starttime = datetime.datetime.now()

# Load the Dataset
csv_data = np.loadtxt(str(path_label),delimiter=',')
labeledData = np.where(csv_data[:,3] == 1)[0]

print('ExtractLabeledFrameDataset.py : Detected {:d} labeled image\n'.format(len(labeledData)))
print('ExtractLabeledFrameDataset.py : Extract labeled frames to the {:s}\n'.format(str(path_export)))

vid = cv.VideoCapture(str(path_video))
zip(np.arange(len(labeledData)), labeledData)

for datanum, framenum in enumerate(labeledData):
    vid.set(cv.CAP_PROP_POS_FRAMES,framenum)
    _, frame = vid.read()
    result = cv.imwrite(str(Path.joinpath(path_export,'{:s}_{:04d}.png'.format(project_name, datanum))),frame)
    if not result:
        raise(Exception('Can not save the frame'))

# Save location file
np.savetxt(str(Path.joinpath(path_export,'{:s}_label.csv'.format(str(path_export)))),csv_data[labeledData,:-1], delimiter='\t')

print('ExtractLabeledFrameDataset.py : Elapsed time : {:.5f} DONE'.format((datetime.datetime.now() - starttime).total_seconds()))
