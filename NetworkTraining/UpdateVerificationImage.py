"""
UpdateVerificationImage.py
@Knowblesse 2021
21 AUG 26
Generate Verification image for data validation
"""

import numpy as np
import cv2 as cv
from pathlib import Path
from checkPreviousDataset import checkPreviousDataset

(dataset_csv, dataset_number) = checkPreviousDataset()
dataset_image = [x for x in sorted(Path('./Dataset').glob('*.png'))]
for i, image_path in enumerate(dataset_image):
    img = cv.imread(str(image_path))
    try:
        img[round(dataset_csv[i,1]), round(dataset_csv[i,2]),:] = 255
    except IndexError as e:
        print(f'UpdateVerificationImage : labeled coordinate of {i:d} image is not in the ROI! Check the image')
        continue
    img = cv.circle(img, (round(dataset_csv[i,2]), round(dataset_csv[i,1])), 3, [0,0,255], -1 )
    cv.line(img, (round(dataset_csv[i,2]), round(dataset_csv[i,1])), (round(dataset_csv[i,2] + 30*np.cos(np.deg2rad(dataset_csv[i,3]))), round(dataset_csv[i,1] + 30*np.sin(np.deg2rad(dataset_csv[i,3])))), [0,255,255], 2)
    cv.imwrite(str(Path(f'./Dataset/Verification/Dataset_verif_{i:03d}.png')), img)

