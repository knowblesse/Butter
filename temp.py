import cv2 as cv
import numpy as np
import random
import time
# ROI selection
box_rat = cv.selectROI('MultiTracker', frame_choice)
cv.waitKey(100)
cv.destroyWindow('MultiTracker')
box_robot = cv.selectROI('MultiTracker', frame_choice)
cv.waitKey(100)
cv.destroyWindow('MultiTracker')

# Get Stat from the ROI
ROI_rat = frame_choice[box_rat[1] : box_rat[1] + box_rat[3], box_rat[0] : box_rat[0]+ box_rat[2], :]
ROI_robot = frame_choice[box_robot[1] : box_robot[1] + box_robot[3], box_robot[0] : box_robot[0]+ box_robot[2], :]

def getBoundary(image, std=1):
    """
    Returns upper boundary and lower boundary
    :param image: image to analyze
    :param std: std variable. default 1
    :return: list : upper, lower
    """
    mu = np.median(image, axis=(0, 1))
    sigma = np.std(image, axis=(0, 1))
    return [mu+(std*sigma), mu-(std*sigma)]

upper_rat, lower_rat = getBoundary(ROI_rat, 0.5)
upper_robot, lower_robot = getBoundary(ROI_robot, 0.5)

# Make Tracker
tracker_rat = cv.TrackerMIL_create()
tracker_robot = cv.TrackerMIL_create()

# Tracker
tracker_rat.init(cv.inRange(frame_choice, lower_rat, upper_rat), box_rat)
tracker_robot.init(cv.inRange(frame_choice,lower_robot, upper_robot), box_robot)

# Process video and track objects
frame_num = 5000
cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
tic = time.time()
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("asdf")
        break

    # Apply color filter
    frame_rat = cv.inRange(frame, lower_rat, upper_rat)
    frame_robot = cv.inRange(frame, lower_robot, upper_robot)

    # get updated location of objects in subsequent frames
    success, box_rat = tracker_rat.update(frame_rat)
    success, box_robot = tracker_robot.update(frame_robot)

    # draw rat
    p1 = (int(box_rat[0]), int(box_rat[1]))
    p2 = (int(box_rat[0] + box_rat[2]), int(box_rat[1] + box_rat[3]))
    cv.rectangle(frame, p1, p2, [255, 0, 0], 2, 1)

    p3 = (int(box_robot[0]), int(box_robot[1]))
    p4 = (int(box_robot[0] + box_robot[2]), int(box_robot[1] + box_robot[3]))
    cv.rectangle(frame, p3, p4, [0, 255, 0], 2, 1)
    point_rat = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    point_robot = ((p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2)
    threshold = 90
    HowFar = np.sqrt((point_rat[0] - point_robot[0]) ** 2 + (point_rat[1] - point_robot[1]) ** 2)
    if HowFar < threshold:
        cv.circle(frame, (100, 100), 5, (0, 0, 255), -1)
    cv.imshow('MultiTracker', frame)
    toc = time.time()
    print(1 / (toc - tic))
    tic = toc

    # quit on ESC button
    if cv.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
cv.destroyWindow('MultiTracker')