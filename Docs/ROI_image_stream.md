## Contents
* [[#Purpose]]
* [[#Method]]
	* [[#1. Read Frame]]
	* [[#2. Apply Global Mask]]
	* [[#3. Background model]]
		* [[#3-1. Building a background model]]
		* [[#3-2. Manually provide a background model]]
	* [[#4. Foreground model]]
		* [[#4-1. Building a foreground model]]
		* [[#4-2. Manually provide a foreground model]]
	* [[#5. Start ROI extraction]]
		* [[#5-1. _ _readVideo]]
		* [[#5-2. _ _processFrames]]
		* [[#5-3. _ _findBlob]]
---
# Purpose
- This file contains a class for extracting a Region Of Interest, a small image patch containing an animal. 
- As the input video is usually too large for the CNN to digest, this script speeds up animal head detection by cropping the frame and extracting the ROI. 
- This function saves blob centers into the queue when called the [[#5. Start ROI extraction]].
- After calling, the `VideoProcessor.py` uses the saved frames and the blob's center point to feed the CNN.

# Method
## 1. Read Frame
- First, the ROI_image_stream must read a **frame** from the video.
- In the best circumstances, this can be achieved by setting the `VideoCapture` object's `CAP_PROP_POS_FRAMES` to desired frame location. 
- However, as OpenCV's [Issue #9053](https://github.com/opencv/opencv/issues/9053) shows, The opencv's video reading object is terrible. 
	- Just never try to track ***Variable Fraem Rate Video***.
- One method that outputs "less" error is going through whole frames by `VideoCapture.grab()` and stopping the desired location and using `VideoCapture.read()` to retrieve the desired frame. 
- As anyone can guess, this method is extremely slow as the length of the video gets longer. 
- Luckily, what a user wants to do with this package is extract *a series of frames* which are aligned by its time. So by using a proper mixture of the `grab()` and the `read()` functions, the script does not need to go through all frames from the beginning whenever it wants to read a frame.  ^bd33f1
- **`ROI_image_stream.getFrame()`** uses the slow method described above, but that function is rarely called and mostly for debugging purposes.

---
## 2. Apply Global Mask
- To decrease error, a global mask can be set. 
- This function is useful as it can exclude any movement outside the apparatus.

---
## 3. Background model
- Butter uses a background model to perform the foreground-background separation.
- Background models can be either automatically built or provided manually.
- The background must be unchanged during the whole video.
  
> [!warning]
> If another highly moving object inside the apparatus has a similar shape to the animal, the foreground-background separation would likely fail.
- If the background remains unchanged through multiple videos, then precalculated background model can be manually fed, and this will decrease total analysis time.
  
### 3-1. Building a background model
* Call `ROI_image_stream.buildBackgroundModel(num_frames2use=200)`
* This function reads 200 (adjustable) equally spaced frames and takes the median frame as the background model.

> [!Warning]
> If the animal stays in an area for a long time, the animal's image could affect the quality of the background model

>[!TODO]
> Include options for using manually selected frames, not equally spaced frames.
+ After, calling `ROI_image_stream.getBackgroundModel()` will return the built background for further use.

### 3-2. Manually provide a background model
* Call `ROI_image_stream.setBackgroundModel(image)`to manually set the background model. 
* The `image` is a 3D numpy array.
 
---
## 4. Foreground model
+ This package assumes any object that moves in the apparatus is highly likely to be a target animal.
  
> [! info]
> Most of tracking software assumes any moving object is the animal. However, Butter takes other parameters of the moving object and selects the object with the highest probability of being the target.

- First, all blobs are extracted, and their size, convexity, and circularity are calculated.
- These values are compared with the foreground model to find the blob with a high chance of being the animal.
- The foreground model can be automatically built or manually provided as in the background model section.

### 4-1. Building a foreground model
* Call `ROI_image_stream.buildForegroundModel(num_frames2use=200)`
* This function reads 200 (adjustable) equally spaced frames and takes the median frame as the background model.
* If `ROI_image_stream.buildBackgroundModel(num_frames2use=200)` was called before, pre-stored sample frames are used to reduce processing time.
> [!info]
> Foreground model is a simple `dict` object containing average 1) size, 2) convexity, and 3)circularity of the target animal.

> [!Warning]
> This function uses the median value for the largest detected blob property. Therefore, if another object larger than the animal frequently appeared in the video, building the foreground model would likely be failed.

+ After, calling `ROI_image_stream.getForegroundModel()` will return the built foreground model for further use.

### 4-2. Manually provide a foreground model
* Call `ROI_image_stream.setForegroundModel(property)` to manually set the background model. 
* The `property` is a dict object with the following keys

---
## 5. Start ROI extraction
- ROI extraction uses multithreading.
- When `ROI_image_stream.startROIextractionThread(start_frame, stride=5)` function is called, it spawn two threads:
	- **Thread1**: `__readVideo` : Read frame from the video
	- **Thread2**: `__processFrames` : Detect Blob
### 5-1. \_\_readVideo
- As mentioned in the [[#^bd33f1|previous block]], this function skips designated start frames and uses a mixture of grab and get to save frames into the `frameQ`.
### 5-2. \_\_processFrames
* This function checks the frame Queue and calls `__findBlob(image, prevPoint=None)` function to find the center position of the blob.
* Every center point of a blob is added into the `blobQ`
### 5-3. \_\_findBlob
* This function uses the following method to find the blob
	1. Set `pastFrameImage` using `cv.addWeighted` function. (0.9 of the previous frame, and 0.1 of the current frame)
	   - This allows smoothed change in `pastFrameImage`
	2. Using `cv.addWeighted` function, it extracts foreground by finding a difference in the current frame compared to the 1) median frame and 2) `pastFrameImage`
	   * By using both median frame and `pastFrameImage`, the foreground extraction can utilize whole-session-wide and time-to-time image differences.
	   * If only whole-session-wide image difference is used to extract foreground, small changes such as urine and feces may be detectd as foreground.
	   * If only time-to-time image difference is used, animal becomes the background when the animal does not move for a while.
	3. Convert the image to grayscale, and use `cv.threshold` function to remove areas darker than the animal.
	4. Call `__denoiseBinaryImage` function to remove small changes and enlarge larger blobs
	5. Find contours and get 1) size, 2) convexity, 3) circularity value for every detected blob.
	6. If the previous location is provided, use that value to add a higher likelihood to the blob closer to the previous location
	7. Find the blob which has the highest possibility of being the target, and return the image center location of that blob
> [!TODO]-
> Use animal color options to include black animals
