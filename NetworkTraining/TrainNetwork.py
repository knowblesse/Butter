from datetime import datetime
from datetime import timedelta
import os
import pickle
import time

from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping

import loadDataset

# Tensorflow version test
if tf.__version__ < '2.5.0':
    print(tf.__version__)
    print("this code is build on tensorflow=2.5.0")
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

################################################################
# Build Model - Base model
################################################################
base_model = keras.applications.MobileNetV2(input_shape=X_conv.shape[1:], include_top=False, weights='imagenet')
base_model.trainable = False

################################################################
# Build Model - Linker model
################################################################
l2_MaxP = layers.MaxPooling2D(pool_size=(3, 3))(base_model.get_layer('block_2_add').output)
l5_ConvT = layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(base_model.get_layer('block_5_add').output)
l5_MaxP = layers.MaxPooling2D(pool_size=(3, 3))(l5_ConvT)
l12_ConvT = layers.Conv2DTranspose(32, kernel_size=(3,3), strides=(4, 4), padding='valid', output_padding=(1, 1))(base_model.get_layer('block_12_add').output)
l12_MaxP = layers.MaxPooling2D(pool_size=(3, 3), padding='valid')(l12_ConvT)
linker_input = keras.layers.concatenate([l2_MaxP, l5_MaxP, l12_MaxP])
linker_output = keras.layers.Flatten()(linker_input)

################################################################
# Build Model - FC model
################################################################

FC = keras.layers.Dropout(0.2, name='FC_DO1')(linker_output)
FC = keras.layers.Dense(500, activation='relu', name='FC_1')(FC)
FC = keras.layers.Dropout(0.2, name='FC_DO2')(FC)
FC = keras.layers.Dense(200, activation='relu', name='FC_2')(FC)
FC = keras.layers.Dense(4, activation='linear',name='FC_3')(FC)

################################################################
# Compile and Train
################################################################

new_model = Model(inputs=base_model.input, outputs=FC)

################################################################
# Callbacks
################################################################

def scheduler(epoch, lr):
    """
    Callback function for adaptive learning rate change
    """
    if epoch < 60:
        return 1e-5
    else:
        return lr * tf.math.exp(-0.01)

learningRateScheduler = LearningRateScheduler(scheduler)
es = EarlyStopping(monitor='val_loss', min_delta=5e-3, patience=30, restore_best_weights=True)

new_model.save_weights('default_weights.h5')

batch_size = 32
momentum = 0.6

model_name = 'Model_' + datetime.now().strftime('%y%m%d_%H%M')
os.mkdir(model_name)
new_model.load_weights('default_weights.h5')

optimizer = keras.optimizers.RMSprop(learning_rate=1e-5, momentum=momentum)
new_model.compile(optimizer=optimizer, loss='mae', metrics='mae')
start_time = time.time()
history = new_model.fit(X_conv,y,epochs=2000, verbose=1, callbacks=[learningRateScheduler, es], validation_split=0.2, batch_size=batch_size)

# Save Model
print('Saving...')
with open('history_' + model_name + '.pickle', 'wb') as f:
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
new_model.save(Path(model_name))
print(f'Saved {model_name}')
print('Elapsed time : ' + str(timedelta(seconds=time.time() - start_time)))
print('Best Score : ' + str(min(history.history['val_mae']))) 

