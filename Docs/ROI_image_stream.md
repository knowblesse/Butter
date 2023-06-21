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

## 3. Foreground-background separation
- Next, foreground-background separation is necessary to detect the animal. 
- This package assumes any object that moves in the apparatus has high possibility of being an target animal. (Not every moving object is detected as the animal!)
### 3-1. Build Background model
- First, 200 (modifiable) equally spaced frame is used to build a background model.
- The model takes the median frame as the background model
> [!Warning]
> If the animal stays in a area for really long time, animal's image could be effect the quality of the background model




