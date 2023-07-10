from datetime import datetime
from datetime import timedelta
import os
import pickle
import time

from pathlib import Path
import numpy as np
from numpy.random import default_rng

import tensorflow as tf
from tensorflow import keras
from Model import createNewButterModelv1

from keras.callbacks import LearningRateScheduler, EarlyStopping, CSVLogger

import loadDataset

# Tensorflow version test
if tf.__version__ != '2.10.0':
    print(tf.__version__)
    print("this code is build on tensorflow=2.10.0")
    raise(BaseException('Tensorflow version mismatch!'))

if tf.test.is_gpu_available():
    print("TF is using GPU")
else:
    print("TF is using CPU")
gpus= tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

################################################################
# Load Dataset
################################################################
# X_conv : converted to match the network input format
#          specifically, processed_data = original_data/127.5-1
X_conv, y = loadDataset.loadDataset()

# Shuffle
rng = default_rng()
shuffled_idx = rng.permutation(list(np.arange(X_conv.shape[0])))

X_conv = X_conv[shuffled_idx, :, :, :]
y = y[shuffled_idx, :]

new_model = createNewButterModelv1(X_conv)

################################################################
# Hyperparameters
################################################################
batch_size = 64
momentum = 0.8
initial_learning_rate = 1e-5
epochs = 100

################################################################
# Make Folder
################################################################
model_name = 'Model_' + datetime.now().strftime('%y%m%d_%H%M%S')
ModelsParentPath = Path(__file__).absolute().parent.parent / 'Models'
if not ModelsParentPath.is_dir():
    raise BaseException("Can not locate 'Models' folder")
ModelPath_str = str((ModelsParentPath / model_name).absolute())
os.mkdir(ModelPath_str)

################################################################
# Callbacks
################################################################
def scheduler(epoch, lr):
    """
    Callback function for adaptive learning rate change
    """
    if epoch < 10:
        return initial_learning_rate
    else:
        return lr * tf.math.exp(-0.01)

learningRateScheduler = LearningRateScheduler(scheduler)
es = EarlyStopping(monitor='val_loss', min_delta=5e-3, patience=10, restore_best_weights=True)
csv_logger = CSVLogger(Path(ModelPath_str) / 'history.csv')

################################################################
# Compile and run
################################################################
optimizer = keras.optimizers.RMSprop(learning_rate=initial_learning_rate, momentum=momentum)
new_model.compile(optimizer=optimizer, loss='mae', metrics='mae')
start_time = time.time()
history = new_model.fit(X_conv,y,epochs=epochs, verbose=1, callbacks=[learningRateScheduler, es, csv_logger], validation_split=0.3, batch_size=batch_size)

################################################################
# Save Model
################################################################
print('Saving...')
new_model.save(ModelPath_str)
print(f'Saved {ModelPath_str}')
print('Elapsed time : ' + str(timedelta(seconds=time.time() - start_time)))
print('Best Score : ' + str(min(history.history['val_mae']))) 

