from tensorflow import keras
from keras.callbacks import LearningRateScheduler, EarlyStopping, CSVLogger
import loadDataset
import time
from pathlib import Path
import numpy as np
from datetime import timedelta
from tkinter.filedialog import askdirectory

################################################################
# Load Model
################################################################
model_path = Path(askdirectory())
model = keras.models.load_model(str(model_path))

history = np.loadtxt(model_path / 'history.csv', skiprows=1, delimiter=',')
last_epoch = history.shape[0]

################################################################
# Load Extra Data
################################################################
X_conv, y = loadDataset.loadDataset()

################################################################
# Hyperparameters
################################################################
batch_size = 32
momentum = 0.8
initial_learning_rate = 1e-8
additional_epochs = 10

################################################################
# Callbacks
################################################################
def scheduler(epoch, lr):
    return initial_learning_rate
learningRateScheduler = LearningRateScheduler(scheduler)
es = EarlyStopping(monitor='val_loss', min_delta=5e-3, patience=50, restore_best_weights=True)
csv_logger = CSVLogger(model_path / 'history.csv', append=True)

################################################################
# Run
################################################################
start_time = time.time()
history = model.fit(X_conv,y,epochs=additional_epochs + last_epoch, verbose=1, initial_epoch=last_epoch, callbacks=[learningRateScheduler, es, csv_logger], validation_split=0.3, batch_size=batch_size)

print('Saving...')
model.save(model_path)
print(f'Saved {model_path}')
print('Elapsed time : ' + str(timedelta(seconds=time.time() - start_time)))
print('Best Score : ' + str(min(history.history['val_mae'])))