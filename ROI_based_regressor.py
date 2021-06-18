import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from helper_function import ROI_image_stream
from helper_function import checkDataSet

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
path_dataset = Path.home() / 'VCF/butter/dataset'
path_csv = next(path_dataset.glob('*.csv'))
base_network = 'mobilenet'

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
    csv_data = np.loadtxt(str(path_csv),delimiter=',')
    labeledIndex = np.where(csv_data[:,3] == 1)[0]
    y_raw = csv_data[labeledIndex, :-1]
except:
    raise(BaseException('Label csv file reading error'))

dataSize = y_raw.shape[0]

################################################################
# Input data : X
################################################################
X = np.zeros((dataSize,base_network_inputsize,base_network_inputsize,3))
ROI_coor = np.zeros((dataSize,2))

istream = ROI_image_stream(path_dataset,ROI_size=base_network_inputsize)
istream.trainBackgroundSubtractor()

for i, frame_number in enumerate(labeledIndex):
    chosen_image, coor = istream.extractROIImage(frame_number)
    X[i, :, :, :] = chosen_image / 255
    ROI_coor[i,:] = coor

in_roi_location = y_raw[:,0:2] - ROI_coor + base_network_inputsize/2 # in_roi_location
y = np.hstack((in_roi_location,np.expand_dims(y_raw[:,2],axis=1)))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#################################################################
# Check loaded Dataset
################################################################
idx = 55
istream.vid.set(cv.CAP_PROP_POS_FRAMES,labeledIndex[idx])
img = istream.vid.read()
img = img[1]

plt.clf()
plt.imshow(img/255)
plt.scatter(y_raw[idx,1], y_raw[idx,0])

r = 30
plt.plot([y_raw[idx,1], y_raw[idx,1] + r*np.cos(np.deg2rad(y_raw[idx,2]))], [y_raw[idx,0], y_raw[idx,0] - r*np.sin(np.deg2rad(y_raw[idx,2]))], LineWidth=3, color = 'r')

print("(%03d, %03d)@%03d" % (int(y_raw[idx,0]), int(y_raw[idx,1]), int(y_raw[idx,2])))



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
    linker1 = keras.layers.add([base_model.get_layer('block_6_project_BN').output, base_model.get_layer('block_9_project_BN').output])
    linker_input = keras.layers.concatenate([final_layer_ConvT, linker1])
    #linker_input = keras.layers.add([final_layer_ConvT, base_model.get_layer('block_9_project_BN').output])
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
FC = keras.layers.Dense(150, activation='selu', name='FC_3')(FC)
FC = keras.layers.Dense(50, activation='selu', name='FC_4')(FC)
FC = keras.layers.Dense(3, activation='linear',name='FC_5')(FC)

################################################################
# Compile and Train
################################################################
new_model = Model(inputs=base_model.input, outputs=FC)


new_model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-5, momentum=0.05), loss='mae', metrics='mae')
#new_model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=1e-4, initial_accumulator_value=0.2), loss='mae', metrics='mae')
history1 = new_model.fit(X_train,y_train,epochs=500,validation_split=0.1,batch_size=20)
plt.clf()
plt.plot(history1.history['loss'])

new_model.evaluate(X_test, y_test)


################################################################
# Test Testing Images
################################################################
y_pred = new_model.predict(X_test)
idx_list = np.random.permutation(y_test.shape[0])
for idx in idx_list[:10]:
    plt.clf()
    plt.imshow(X_test[idx,:,:,:])
    # draw predicted
    r = 40
    plt.scatter(y_pred[idx,1], y_pred[idx,0],c='r')
    plt.plot([y_pred[idx,1], y_pred[idx,1] + r*np.cos(np.deg2rad(y_pred[idx,2]))], [y_pred[idx,0], y_pred[idx,0] - r*np.sin(np.deg2rad(y_pred[idx,2]))], lineWidth=3, color = 'r')
    # draw real
    plt.scatter(y_test[idx,1], y_test[idx,0],c='g')
    plt.plot([y_test[idx,1], y_test[idx,1] + r*np.cos(np.deg2rad(y_test[idx,2]))], [y_test[idx,0], y_test[idx,0] - r*np.sin(np.deg2rad(y_test[idx,2]))], lineWidth=3, color = 'g')
    # print output
    print("%05d : (%03d, %03d)@%03d : (%03d, %03d)@%03d" % (int(idx), int(y_test[idx,0]), int(y_test[idx,1]), int(y_test[idx,2]), int(y_pred[idx,0]), int(y_pred[idx,1]), int(y_pred[idx,2])))
    plt.pause(1)

new_model.save('210617_Target_model')

################################################################
# Test New Images 
################################################################

idx_list = 5000 + np.random.permutation(5000)
for idx in idx_list[:5]:
    chosen_image, coor = istream.extractROIImage(idx)
    testing = np.expand_dims(chosen_image,0) / 255

    result = new_model.predict(testing)
    plt.clf()
    plt.imshow(chosen_image/255)
    plt.scatter(result[0,1], result[0,0])
    r = 20
    plt.plot([result[0,1], result[0,1] + r*np.cos(np.deg2rad(result[0,2]))], [result[0,0], result[0,0] - r*np.sin(np.deg2rad(result[0,2]))], lineWidth=3, color = 'r')
    print("%05d : (%03d, %03d)@%03d" % (int(idx), int(result[0,0]), int(result[0,1]), int(result[0,2])))
    plt.pause(1)

