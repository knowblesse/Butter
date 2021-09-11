# Butter
Goes well with Lobster

# File Lists

## root

**ProcessVideo.py** : Process a new video file with pretrained neural net

**istream_test.py** : testing script for parameter optimization

## Network Training

**/NetworkTraining/checkPreviousDataset** : Dataset integrity checking function used for appending dataset for network training

**/NetworkTraining/GenerateTrainingDataset** : Extract ROI image data from labeled video and merge to the dataset

**/NetworkTraining/TrainNetwork** : TrainNetwork with current dataset

**/NetworkTraining/UpdateVerificationImage** : Using the dataset, generated marked images for verification purpose

# Notes

## VideoProcessor
**CPU** : Intel Core i7-6700 CPU (3.40GHz)

**Testing Data** : 10934 frames (640 x 480)

* Single thread
  * 07:56 = 22.97fps
* Separated Video I/O thread
  * 07:08 = 25.54fps
* Separated Vid I/O thread and opencv process thread
  * 06:24 = 28.47fps

## Training

**GPU** : NVIDIA GeForce GTX 1060 3Gb

### Model_210911_6000epoch

**Training Data** : 560 labeled data -> augmented x4

**Hyper parameters**
* lr 5e-7
* mementum 0.07
* loss mae
* batch size 10
* 6000 epochs
* elapsed time : 9:19:26




