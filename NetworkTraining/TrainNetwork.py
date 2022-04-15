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
# Constants
################################################################
base_network = 'mobilenet_v2'
model_name = 'Model_' + datetime.now().strftime('%y%m%d_%H%M')
os.mkdir(model_name)

################################################################
# Load Dataset
################################################################
# X_conv : converted to match the network input format
#          specifically, processed_data = original_data/127.5-1
X_conv, y = loadDataset.loadDataset(base_network)
y = y[:,0:2] # temporal fix to disable head direction detection

################################################################
# Build Model - Base model
################################################################
if base_network == 'mobilenet_v2':
    base_model = keras.applications.MobileNetV2(input_shape=X_conv.shape[1:], include_top=False, weights='imagenet')
elif base_network == 'inception_v3':
    base_model = keras.applications.InceptionV3(input_shape=X_conv.shape[1:], include_top=False, weights='imagenet')
else:
    raise(BaseException('Not implemented'))

base_model.trainable = False

################################################################
# Build Model - Linker model
################################################################
if base_network == 'mobilenet_v2':
    #final_layer_ConvT = layers.Conv2DTranspose(64,kernel_size=(3,3), strides=(2,2), padding='same')(base_model.get_layer('block_14_add').output)
    #linker_input = keras.layers.concatenate([final_layer_ConvT, base_model.get_layer('block_8_add').output])
    lower_layer_ConvT = layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(base_model.get_layer('block_5_add').output)
    linker_input = keras.layers.concatenate([lower_layer_ConvT, base_model.get_layer('block_2_add').output])
    linker_output = keras.layers.Flatten()(linker_input)
elif base_network == 'inception_v3':
    final_layer_ConvT = keras.layers.Conv2DTranspose(2048,kernel_size=(12,12))(base_model.get_layer('mixed5').output)
    linker_input = keras.layers.concatenate([final_layer_ConvT, base_model.get_layer('mixed10').output])
    linker_output = keras.layers.Flatten()(linker_input)
else:
    raise(BaseException('Not Implemented'))

################################################################
# Build Model - FC model
################################################################

FC = keras.layers.Dropout(0.3, name='FC_DO1')(linker_output)
FC = keras.layers.Dense(500, activation='relu', name='FC_1')(FC)
FC = keras.layers.Dropout(0.3, name='FC_DO2')(FC)
FC = keras.layers.Dense(300, activation='relu', name='FC_2')(FC)
FC = keras.layers.Dense(2, activation='linear',name='FC_3')(FC)

################################################################
# Callbacks
################################################################

def scheduler(epoch, lr):
    """
    Callback function for adaptive learning rate change
    """
    if epoch < 300:
        base_model.trainable = False
        return 1e-5
    else:
        if epoch > 400:
            base_model.trainable = True
            # after changing trainable state of the model, you must compile again!
            new_model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-5, momentum=0.3), loss='mae', metrics='mae')
        return lr * tf.math.exp(-0.01)

es = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=10)

################################################################
# Compile and Train
################################################################

learningRateScheduler = LearningRateScheduler(scheduler)
new_model = Model(inputs=base_model.input, outputs=FC)
new_model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-5, momentum=0.3), loss='mae', metrics='mae')
new_model.save_weights(model_name+'_weights.h5')

batch_size = [32, 64, 128, 256, 512]
for b in batch_size:
    new_model.load_weights(model_name+'_weights.h5')
    start_time = time.time()
    save_interval = 100
    total_epoch = 2000
    history_file_name = datetime.now().strftime("%y%m%d_%H%M")
    history = {'loss':[], 'mae': [], 'val_loss': [], 'val_mae': [], 'lr': []}
    for i in np.arange(int(total_epoch/save_interval)):
        hist = new_model.fit(X_conv,y,epochs=int(save_interval*(i+1)),initial_epoch=int(save_interval*i), callbacks=[learningRateScheduler, es], validation_split=0.2,batch_size=b)
        for k in history.keys():
            history[k].extend(hist.history[k])
        print('Saving...')
        with open('history_' + model_name + '.pickle', 'wb') as f:
            pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
        new_model.save(Path(model_name))
        print(f'Saved until {save_interval*(i+1):d} epochs')
    print('Elapsed time : ' + str(timedelta(seconds=time.time() - start_time)))
