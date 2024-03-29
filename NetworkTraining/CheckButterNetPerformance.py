"""
CheckButterNetPerformance
@Knowblesse 2022
Test ButterNet Performance using the current Dataset.
Warning : Since the neural network is trained with the same or mostly same data, the loss will be must lower than
the result from the validation data. Use the metric with care.
"""
import numpy as np
import os

from pathlib import Path
from tensorflow import keras
from tqdm import tqdm
from keras import metrics

from loadDataset import loadDataset

model_path = Path('/home/ubuntu/VCF/butter/NetworkTraining/Model_220504_1730')
try:
    model = keras.models.load_model(str(model_path))
except:
    raise (BaseException('Can not load model from ' + str(model_path)))
print(Path('../NetworkTraining/Dataset/Dataset.csv').absolute())
(X_conv, y) = loadDataset()


y_pred = []
done = False
idx = 0
stride = 200

for i in tqdm(range(int(np.ceil(X_conv.shape[0]/stride)))):
    if idx+stride >= X_conv.shape[0]:
        result = model.predict(X_conv[idx:, :, :, :])
        done = True
    else:
        result = model.predict(X_conv[idx:idx + stride, :, :, :])
    y_pred.extend(result)
    idx += stride

y_pred = np.array(y_pred)

loss = metrics.mean_absolute_error(y, y_pred)
print(f'Model Loss (mae) : {np.round(np.mean(loss),2):.2f}')
print(f'Model Loss (mae) on last 10% data : {np.round(np.mean(loss[int(loss.shape[0]*0.9):]),2):.2f}')
