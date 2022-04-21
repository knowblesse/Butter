# Butter

Automatic, CPU and GPU based animal head tracker

## Why butter?

1. **Superfast.**

    - The CPU detects a blob which has the highest chance of being the animal using a log-likelihood method with the four parameters: size, convexity, circularity, and previous location.
    - A small patch of image around the blob from the CPU thread is then feed into the GPU.
    - The GPU uses a custom deep CNN, butterNet, to detect animal's head and the head direction inside the image patch.
    - Since the video processing (extracting the frame, detecting animal's location) and the image processing (detecting animal's head and head direction) is separated, the processing speed is remarkable.
    
2. **Extremely Accurate.**
	- There are multiple commercial softwares out there specifically built for animal detection. (ANY-Maze, Ethovision XT, etc...)
	- They have user-friendly GUI and has countless functions, but they are really inaccurate when detecting the exact position of the animal's head. (If you are okay with the center of the animal, then just stop reading and use that softwares. 
3. **Don't need training.**
	- Deeplabcut has the highest accuracy like **butter**, but it needs training and maybe bit hard to setup.
	- If the video clip is not that big and you don't care about the training, use Deeplabcut instead of this package.
4. **Does not require hardcore computer**
	- I tested on the setup below
		- **CPU** : Intel Core i7-6700 CPU (3.40GHz) **GPU** : NVIDIA GeForce GTX 1060 3Gb
		- **Video** : 30m 8s(640 x 480)
		- **Time** : **2m 6s**

5. **Goes well with Lobster**
    


# Setup
1. Clone this repository
2. Download the lastest version of the butterNet from the following link
    [https://cloud.knowblesse.com/sharing/pclv8HpwW](https://cloud.knowblesse.com/sharing/pclv8HpwW)
3. Edit the main.py script with proper paths for the video and the model
4. (Optional) Edit the buttered (head tracked file) file with the /util/ReLabeler.py 


# File Lists

## root

**main.py** : Sample main script

**VideoProcessor.py** : Head Detection stream class (incl. ROI_image_stream)

**ROI_image_stream.py** : Animal blob detection stream class


## Network Training

**/NetworkTraining/checkPreviousDataset** : Dataset integrity checking function used for appending dataset for network training

**/NetworkTraining/GenerateTrainingDataset** : Extract ROI image data from labeled video and merge to the dataset

**/NetworkTraining/TrainNetwork** : TrainNetwork with current dataset

**/NetworkTraining/UpdateVerificationImage** : Using the dataset, generated marked images for verification purpose


## util
**/util/ReLabeler.py** : Manual Relabeling App 
**/util/SaveLabeledVideo.py** : Generate Head Location / Head Direction labeled video


updated at version 0.2 (2022 APR 21) Knowblesse

