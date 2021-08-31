import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import time

# Tensorflow version test
if tf.__version__ < '2.5.0':
    print(tf.__version__)
    print("this code is build on tensorflow=2.5.0")
    raise(BaseException('Tensorflow version mismatch!'))

if tf.test.is_gpu_available():
    print("TF is using GPU")
else:
    print("TF is using CPU")
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

################################################################
# Constants
################################################################
base_network = 'mobilenet'
model_save_path = Path('./Model_210828_2000epoch')
################################################################
# Setup
################################################################
if base_network == 'mobilenet':
    base_network_inputsize = 224
elif base_network == 'inception_v3':
    base_network_inputsize = 300
else:
    raise(BaseException('Not implemented'))

################################################################
# Input data : y
################################################################
try:
    csv_data = np.loadtxt(str(Path('./Dataset/Dataset.csv')),delimiter=',')
    y_raw = csv_data[:,1::]
except:
    raise(BaseException('Label csv file reading error'))
dataSize = y_raw.shape[0]

################################################################
# Input data Processing
################################################################
X = np.zeros((4 * dataSize,base_network_inputsize,base_network_inputsize,3))
y = np.zeros((4 * dataSize,4))

dataset_image = [x for x in sorted(Path('./Dataset').glob('*.png'))]

if dataSize != len(dataset_image):
    raise(BaseException('TrainNetwork : Dataset size mismatch'))

# Data augmentation
for i, clip in enumerate(dataset_image):
    chosen_image = cv.imread(str(clip))
    X[i * 4 + 0, :, :, :] = chosen_image / 255
    X[i * 4 + 1, :, :, :] = cv.flip(chosen_image, 0) / 255 # updown (row)
    X[i * 4 + 2, :, :, :] = cv.flip(chosen_image, 1) / 255 # leftright (col)
    X[i * 4 + 3, :, :, :] = cv.flip(chosen_image, -1) / 255 # both

    corr = y_raw[i, 0:2]
    y[i * 4 + 0, 0:2] = corr
    y[i * 4 + 1, 0:2] = [base_network_inputsize - corr[0], corr[1]] #updown (row)
    y[i * 4 + 2, 0:2] = [corr[0], base_network_inputsize - corr[1]] #leftright (col)
    y[i * 4 + 3, 0:2] = [base_network_inputsize - corr[0], base_network_inputsize - corr[1]]

    # Degree coding into 30 pixel away point
    r = 30
    y[i * 4 + 0, 2:4] = [
            y[i * 4 + 0, 0] + r*np.sin(np.deg2rad(y_raw[i,2])),
            y[i * 4 + 0, 1] + r*np.cos(np.deg2rad(y_raw[i,2]))]
    y[i * 4 + 1, 2:4] = [
            y[i * 4 + 1, 0] - r*np.sin(np.deg2rad(y_raw[i,2])),
            y[i * 4 + 1, 1] + r*np.cos(np.deg2rad(y_raw[i,2]))]
    y[i * 4 + 2, 2:4] = [
            y[i * 4 + 2, 0] + r*np.sin(np.deg2rad(y_raw[i,2])),
            y[i * 4 + 2, 1] - r*np.cos(np.deg2rad(y_raw[i,2]))]
    y[i * 4 + 3, 2:4] = [
            y[i * 4 + 3, 0] - r*np.sin(np.deg2rad(y_raw[i,2])),
            y[i * 4 + 3, 1] - r*np.cos(np.deg2rad(y_raw[i,2]))]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#################################################################
# Check loaded Dataset
#################################################################

idx = np.random.randint(dataSize*4)

plt.clf()
plt.imshow(X[idx,:,:,:])
plt.scatter(y[idx,1], y[idx,0])

r = 30
plt.plot([y[idx,1], y[idx,3]], [y[idx,0], y[idx,2]], LineWidth=3, color = 'r')

################################################################
# Build Model - Base model
################################################################
if base_network == 'mobilenet':
    base_model = keras.applications.MobileNetV2(input_shape=X.shape[1:], include_top=False, weights='imagenet')
elif base_network == 'inception_v3':
    base_model = keras.applications.InceptionV3(input_shape=X.shape[1:], include_top=False, weights='imagenet')
else:
    raise(BaseException('Not implemented'))

base_model.trainable = False

################################################################
# Build Model - Linker model
################################################################
if base_network == 'mobilenet':
    final_layer_ConvT = layers.Conv2DTranspose(64,kernel_size=(8,8))(base_model.get_layer('block_16_project_BN').output)
    linker_input = keras.layers.add([final_layer_ConvT, base_model.get_layer('block_9_project_BN').output])
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
FC = keras.layers.Dense(200, activation='selu', name='FC_1')(linker_output)
FC = keras.layers.Dropout(0.3, name='FC_DO')(FC)
FC = keras.layers.Dense(150, activation='selu', name='FC_2')(FC)
FC = keras.layers.Dense(4, activation='linear',name='FC_3')(FC)

################################################################
# Compile and Train
################################################################
new_model = Model(inputs=base_model.input, outputs=FC)
new_model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-6, momentum=0.05), loss='mae', metrics='mae')
#new_model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=1e-4, initial_accumulator_value=0.2), loss='mae', metrics='mae')
start_time = time.time()
save_interval = 100
total_epoch = 3000
for i in np.arange(int(total_epoch/save_interval)):
    new_model.fit(X_train,y_train,epochs=int(save_interval*(i+1)),initial_epoch=int(save_interval*i), validation_split=0.1,batch_size=20)
    print('Saving...')
    new_model.save(model_save_path)
    print(f'Saved until {save_interval*(i+1):d} epochs')
print('Elapsed time : ' + str(datetime.timedelta(seconds=time.time() - start_time)))

################################################################
# Test with testset images
################################################################
y_pred = new_model.predict(X_test)
idx_list = np.random.permutation(y_test.shape[0])
for idx in idx_list[:10]:
    plt.clf()
    plt.imshow(X_test[idx,:,:,:])
    # draw predicted
    r = 40
    plt.scatter(y_pred[idx,1], y_pred[idx,0],c='r')
    plt.plot([y_pred[idx,1], y_pred[idx,3]], [y_pred[idx,0], y_pred[idx,2]],lineWidth=3, color = 'r')
    # draw real
    plt.scatter(y_test[idx,1], y_test[idx,0],c='g')
    plt.plot([y_test[idx,1], y_test[idx,3]], [y_test[idx,0], y_test[idx,2]], lineWidth=3, color = 'g')
    # print output
    plt.pause(1)

