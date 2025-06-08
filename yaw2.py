

import cv2
import numpy as np
import time

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Process two images
index = 2497
img_path1 = "imgdatapool/img" + str(index) + ".jpg"

prev_frame = cv2.imread(img_path1)
# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


# Detect corners to track in the first frame
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)


net_angle_diff = 0
while index < 2592:
    index += 1
    print(index)

    img_path2 = "imgdatapool/img" + str(index) + ".jpg"

    curr_frame = cv2.imread(img_path2)

    # Convert the current frame to grayscale
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    curr_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    # Select good points
    good_new = curr_pts[status == 1]
    good_old = prev_pts[status == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        cv2.line(curr_frame, (a, b), (c, d), (0, 255, 0), 2)
        cv2.circle(curr_frame, (a, b), 5, (0, 0, 255), -1)

    # Display the result
    cv2.imshow('Optical Flow', curr_frame)

    # Update the previous frame and points
    prev_gray = curr_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)
    
    # Exit on ESC key
    if cv2.waitKey(45) & 0xFF == 27:
        break

print("Net Yaw: ", net_angle_diff)
cv2.destroyAllWindows()

