"""
checkPreviousDataset.py
@Knowblesse 2021
21 AUG 26
Function used for appending dataset for network training.
Check the integrity of the previous dataset and return dataset with index for new data
If user intentionally delete some pictures, then corresponding data in .csv file is erased and all dataset is rearranged.
- Output
    - dataset_csv : loaded dataset csv
    - dataset_number : dataset index to begin
"""
import numpy as np
from pathlib import Path
import re

def checkPreviousDataset(datasetLocation = Path('./Dataset/'), silent=False):
    if datasetLocation.is_dir(): # Dataset file exist
        # Check data number from CSV
        dataset_csv = np.loadtxt(str(next(datasetLocation.glob('*.csv'))), delimiter=',')

        # Check data number from Image
        dataset_image = [str(x.name) for x in sorted(datasetLocation.glob('*.png'))]

        # Compare size of  csv and image
        if len(dataset_image) == dataset_csv.shape[0]:
            if not silent:
                print('checkPreviousDataset : Dataset size match')
        elif len(dataset_image) < dataset_csv.shape[0]:
            if not silent:
                print('checkPreviousDataset : Some of images are deleted! deleting corresponding csv data entries...')
        elif len(dataset_image) > dataset_csv.shape[0]:
            raise(BaseException('checkPreviousDataset : Some of csv data entries are missing!'))

        # Check Data order
        image_num = [int(re.search('(\d\d\d\d)', x).group()) for x in dataset_image]
        missing_image_number = []
        for i in np.arange(dataset_csv.shape[0]):
            if not(i in image_num):
                missing_image_number.append(i)

        # Delete csv log of missing image
        dataset_csv = np.delete(dataset_csv,missing_image_number,axis=0)

        # Relabel CSV data entries
        dataset_csv[:,0] = np.arange(dataset_csv.shape[0])
        np.savetxt(datasetLocation/'Dataset.csv', dataset_csv, delimiter=',')
        dataset_number = dataset_csv.shape[0]

        # Relabel Image
        for i, path in enumerate(sorted(datasetLocation.glob('*.png'))):
            path.rename(datasetLocation / Path(f'Dataset_{i:04d}.png'))
        if not silent:
            print(f'checkPreviousDataset : {dataset_number} data is confirmed')
            print('checkPreviousDataset : New Data will be appended to this data')
    else:
        dataset_csv = np.zeros((0,4))
        dataset_number = 0
        if not silent:
            print('checkPreviousDataset : New dataset is generated')
    return (dataset_csv, dataset_number)
