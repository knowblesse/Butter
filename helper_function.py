import numpy as np
import matplotlib.pyplot as plt

################################################################
# Check Dataset New Images
################################################################
def checkDataSet(X,y):
    for idx in np.arange(X.shape[0]):
        r = 20
        plt.clf()
        plt.imshow(X[idx,:,:,:])
        # draw real
        plt.scatter(y[idx,1], y[idx,0],c='g')
        plt.plot([y[idx,1], y[idx,1] + r*np.cos(np.deg2rad(y[idx,2]))], [y[idx,0], y[idx,0] - r*np.sin(np.deg2rad(y[idx,2]))], lineWidth=3, color = 'g')
        plt.title(str(idx))
        # print output
        #print("%05d : (%03d, %03d)@%03d : (%03d, %03d)@%03d" % (int(idx), int(y_test[idx,0]), int(y_test[idx,1]), int(y_test[idx,2]), int(y_pred[idx,0]), int(y_pred[idx,1]), int(y_pred[idx,2])))
        plt.pause(0.3)
