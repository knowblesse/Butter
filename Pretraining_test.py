import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2 as cv
import matplotlib.pyplot as plt
import time
from pathlib import Path

if tf.__version__ < '2.5.0':
    print("this code is build on tensorflow=2.5.0")
    raise(BaseException('Tensorflow version mismatch!'))


if tf.test.is_gpu_available():
    print("TF is using GPU")
else:
    print("TF is using CPU")

# Parameter
path_dataset = Path.joinpath(Path.home(), 'VCF/butter/dataset')

# Paths 
path_image = [paths for paths in sorted((path_dataset/'images').glob('*.png'))]
path_csv = path_dataset.glob('*.csv')

# Generate Dataset
dataSize = len(path_image)
frameSize = cv.imread(str(path_image[0])).shape

# Input data
# TODO : regression data loading from the disk
X = image_dataset_from_directory(path_dataset / 'images', labels=None, image_size=(224, 224))
try:
    y = np.loadtxt(str(next(path_csv)))
except:
    raise(BaseException('Label csv file reading error'))

# Build Model
base_model = keras.applications.MobileNetV2(input_shape=X.element_spec.shape[1:], include_top=False, weights='imagenet')

base_model.trainable = False

base_model.get_layer('block_16_project_BN')

l1 = keras.layers.Conv2DTranspose(96,kernel_size=(8,8))(base_model.get_layer('block_16_project_BN').input)
l2 = keras.layers.add([l1, base_model.get_layer('block_12_project_BN').output])
l3 = keras.layers.Flatten()(l2)
l4 = keras.layers.Dense(100)(l3)
l5 = keras.layers.Dense(30)(l4)
l6 = keras.layers.Dense(3)(l5)

new_model = keras.Model(inputs = base_model.input, outputs=l6)
new_model.compile(optimizer='Adam', loss='mse', metrics='mse')
