from tensorflow import keras
import loadDataset
import time
from pathlib import Path
import numpy as np
import datetime

save_interval = 300
total_epoch = 3000
original_epoch = 900

model_path = Path('./Model_220321')
base_network = 'mobilenet_v2'
X_conv, y = loadDataset.loadDataset(base_network)
new_model = keras.models.load_model(str(model_path))

start_time = time.time()

for i in np.arange(int(total_epoch/save_interval)):
    new_model.fit(X_conv,y,epochs=int(original_epoch + save_interval*(i+1)),initial_epoch=int(original_epoch + save_interval*i), validation_split=0.1,batch_size=10)
    print('Saving...')
    new_model.save(str(model_path)+'_'+str(int(save_interval*(i+1))))
    print(f'Saved until {save_interval*(i+1):d} epochs')
print('Elapsed time : ' + str(datetime.timedelta(seconds=time.time() - start_time)))
