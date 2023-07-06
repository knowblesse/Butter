 
---
# Purpose
- Main class for processing video file.

---
# Method
- Create `Butter` object.
- To run the butter, following information is needed
	- Path to the model
	- Path to the video
	- Background model (either built or provided)
	- Foreground model (either built or provided)
	- Global Mask (optional, but highly recommended)
	- Start Frame (where animal appears first)
- The first two is loaded when creating the object.
	  `butter = Butter(video_path, model_path)`
### Global Mask (optional, but highly recommended)
- Call `Butter.setGlobalMask()`

### Background model
  - Call `Butter.buildBackgroundModel()` or provide with `Butter.setBackgroundModel()`
### Foreground model
  - Call `Butter.buildForegroundModel()` or provide with `Butter.setForegroundModel()`

### Start Position
- Call `Butter.checkStartPosition()`
### Run
- Call `Butter.run()`
### Save
- Call `Butter.save()`

  




