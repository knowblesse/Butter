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
project_name = 'Dev_LARGE'
path_video = Path('/mnt/Data/Data/Small Experiment/LARGE/2021-04-22 11-47-11.mkv')
path_label = Path('/mnt/Data/Data/Small Experiment/LARGE/2021-04-22 11-47-11.mkv_data.csv')
path_export = Path.joinpath(Path.home(),'VCF/butter/dataset')

starttime = datetime.datetime.now()

def acquireVideoCapture():
    global vid
    vid = cv.VideoCapture(str(path_video))

def saveFrame(datanum, framenum):
    #vid = cv.VideoCapture(str(path_video))
    vid.set(cv.CAP_PROP_POS_FRAMES,framenum)
    _, frame = vid.read()
    result = cv.imwrite(str(Path.joinpath(path_export,'{:s}_{:04d}.png'.format(project_name, datanum))),frame)
    if not result:
        raise(Exception('Can not save the frame'))

# Load the Dataset
csv_data = np.loadtxt(str(path_label),delimiter=',')
labeledData = np.where(csv_data[:,3] == 1)[0]

print('ExtractLabeledFrameDataset.py : Detected {:d} labeled image\n'.format(len(labeledData)))
print('ExtractLabeledFrameDataset.py : Extract labeld frames to the {:s}\n'.format(str(path_export)))

# Extract labeled frames using Multiprocessing
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(),initializer=acquireVideoCapture)
pool.starmap(saveFrame,zip(np.arange(len(labeledData)), labeledData))
pool.close()

# Save location file
np.savetxt(str(Path.joinpath(path_export,'{:s}_label.csv'.format(str(path_export)))),csv_data[labeledData,:-1], delimiter='\t')


print('ExtractLabeledFrameDataset.py : Elapsed time : {:.5f} DONE'.format((datetime.datetime.now() - starttime).total_seconds()))




#
#
#
#
#
# fig = plt.figure(1)
# ax = fig.subplots(1,1)
# ax.imshow(frame)
# ax.scatter(csv_data[labeledData[0],0], frameSize[0]-csv_data[labeledData[0],1])
# # remember that array structure and image sturcture has inverse y-axis = row
#





