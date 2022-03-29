import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler
from pathlib import Path
import datetime
import time
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
model_save_path = Path('./Model_220323')

X_conv, y = loadDataset.loadDataset(base_network) # X_conv : converted to match the network input format

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
    final_layer_ConvT = layers.Conv2DTranspose(64,kernel_size=(3,3), strides=(2,2), padding='same')(base_model.get_layer('block_14_add').output)
    linker_input = keras.layers.add([final_layer_ConvT, base_model.get_layer('block_8_add').output])
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
FC = keras.layers.Dropout(0.2, name='FC_DO1')(linker_output)
FC = keras.layers.Dense(200, activation='relu', name='FC_1')(FC)
FC = keras.layers.Dropout(0.2, name='FC_DO2')(FC)
FC = keras.layers.Dense(200, activation='relu', name='FC_2')(FC)
FC = keras.layers.Dense(4, activation='linear',name='FC_3')(FC)

################################################################
# Compile and Train
################################################################
def scheduler(epoch, lr):
    if epoch < 4000:
        base_model.trainable = False
        return 1e-4
    else:
        if epoch > 4500:
            base_model.trainable = True
        return lr * tf.math.exp(-0.01)

learningRateScheduler = LearningRateScheduler(scheduler)
new_model = Model(inputs=base_model.input, outputs=FC)
new_model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-4, momentum=0.1), loss='mae', metrics='mae')
start_time = time.time()
save_interval = 100
total_epoch = 5000
for i in np.arange(int(total_epoch/save_interval)):
    history = new_model.fit(X_conv,y,epochs=int(save_interval*(i+1)),initial_epoch=int(save_interval*i), callbacks=[learningRateScheduler], validation_split=0.2,batch_size=10)
    print('Saving...')
    np.savetxt('history_'+str(i)+'.txt', history.history, delimiter=',')
    new_model.save(model_save_path)
    print(f'Saved until {save_interval*(i+1):d} epochs')
print('Elapsed time : ' + str(datetime.timedelta(seconds=time.time() - start_time)))

#7800 epochs : 15 hours
