import numpy as np
from pathlib import Path
from NetworkTraining.loadDataset import loadDataset



import tensorflow as tf
from tensorflow import keras
#model_path = Path('/home/knowblesse/VCF/butter/Models/Model_220323')
model_path = Path('/home/knowblesse/VCF/butter/Models/butterNet_V1')
try:
    model = keras.models.load_model(str(model_path))
except:
    raise (BaseException('VideoProcessor : Can not load model from ' + str(model_path)))

(X_conv, y) = loadDataset()

y_pred = []
done = False
idx = 0
stride = 20
while not done:
    if idx+stride >= X_conv.shape[0]:
        result = model.predict(X_conv[idx:, :, :, :])
        done = True
    else:
        result = model.predict(X_conv[idx:idx + stride, :, :, :])
    y_pred.extend(result)
    idx += stride
    print(idx)

y_pred = np.array(y_pred)

from keras import metrics
#loss = metrics.mean_absolute_error(y[:,:2], y_pred[:,:2])
loss = metrics.mean_absolute_error(y, y_pred)
print(np.round(np.mean(loss),2))
