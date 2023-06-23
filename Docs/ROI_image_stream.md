# Purpose
- This file contains a class for extracting a Region Of Interest, a small patch of image containing an animal. 
- As the input video usually have too large size for the CNN to digest, this script can speed up animal head detection by cutting the important part of the video. 

# Method
## 1. Read Frame
- First, the ROI_image_stream must read a **frame** from the video.
- In the best circumstatnces, this can be acheived by setting the `VideoCapture` object's `CAP_PROP_POS_FRAMES` to desired frame location. 
- However, as the [Issue #9053](https://github.com/opencv/opencv/issues/9053) shows, The opencv's video reading object is terrible. 
	- Just never try to track ***Variable Fraem Rate Video***.
- One method which output "less" error is going through whole frames by `VideoCapture.grab()` and stopping the desired location and use `VideoCapture.read()` to retrieve the desired frame. 
- As anyone can guess, this method is extremly slow as the length of the video gets longer. 
- Luckly, what user want to do with this package is extract *series of frames* which is aligned by it's time. So by using the proper mixutre of the `grab()` and the `read()` functions, the frame reading header does not have to return to the zero everytime reading a frame. 
- **`ROI_image_stream.getFrame()`** uses the slow method described above, but that function is rarely called and mostly for the debugging purpose.

## 2. Apply Global Mask
- To decrease error, a global mask can be set. 
- This is really usuful function as it can exclude any movement in outside of the apparatus.

---
## 3. Background model
- Butter uses a background model to perfrom foreground-background separation.
- Background model can be either automatically built or provided manually.
- The background must be unchanged during the whole video.
  
> [!warning]
> If there is other highly moving object inside the apparatus and has similar shape as the animal, the foreground-background separation would likely be failed.
- If the background remain unchanged during multiple video, than precalculated background model can be manually feed, and this will decrease total analysis time.
  
### 3-1. Building a background model
* Call `ROI_image_stream.buildBackgroundModel(num_frames2use=200)`
* This function reads 200 (modifiable) equally spaced frame and takes the median frame as the background model.

> [!Warning]
> If the animal stays in a area for really long time, animal's image could be effect the quality of the background model

+ After, calling `ROI_image_stream.getBackgroundModel()` will return the built background for further use.

### 3-2. Manually provide a background model
* Call `ROI_image_stream.setBackgroundModel(image)`
* Using this function with manually set the background model. 
* The `image` is a numpy array with 3D
 
---
## 4. Foreground model
+ This package assumes any object that moves in the apparatus has high possibility of being an target animal.
  
> [! info]
> Most of the tracking software assume any moving object is the animal. But Butter take other parameter of the moving object, and select the object with has the highest probability of being the target animal.

- First, all blobs are extracted, and their size, convexity, and circularity are caculated and these values are compared with the foreground model to find the blob that has the high chance of being the animal.
- As in the background model section, the foreground model can be either automatically built or manually provided.

### 4-1. Building a foreground model
* Call `ROI_image_stream.buildForegroundModel(num_frames2use=200)`
* This function reads 200 (modifiable) equally spaced frame and takes the median frame as the background model.
* If `ROI_image_stream.buildBackgroundModel(num_frames2use=200)` was called before, pre-stored sample frames are used to reduce processing time.
> [!info]
> Foreground model is a simple data showing average 1) size, 2) convexity, and 3)circularity of the target animal.

> [!Warning]
> This function uses the median value for property of the largest detected blob. Therefore, if other object, larger than the animal is frequenty appeared in the video, building the foreground model would likely to be failed.

+ After, calling `ROI_image_stream.getForegroundModel()` will return the built foreground model for further use.

### 4-2. Manually provide a foreground model
* Call `ROI_image_stream.setForegroundModel(property)`
* Using this function with manually set the background model. 
* The `property` is a dict object with following keys


