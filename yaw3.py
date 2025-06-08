import cv2
import numpy as np
import copy


def rotational_to_euler(R):
    if np.isclose(R[2, 0], -1.0):
        pitch = np.pi / 2
        yaw = np.arctan2(R[0, 1], R[0, 2])
        roll = 0.0
    elif np.isclose(R[2, 0], 1.0):
        pitch = -np.pi / 2
        yaw = np.arctan2(-R[0, 1], -R[0, 2])
        roll = 0.0
    else:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    
    return yaw, pitch, roll

#camera matrix of the realsence camera
K = np.array([[675.537322,0.000000,311.191300],
                          [0.000000,677.852071,221.610964],
                          [0, 0, 1]])



# Default Parameters for Shi-Tomasi corner detection
# feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Default Parameters for Lucas-Kanade optical flow
# lk_params = dict(winSize=(15, 15), maxLevel=2,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Process two images
index = 2497
img_path1 = "../imgdatapool/img" + str(index) + ".jpg"

prev_frame = cv2.imread(img_path1)
# Convert the first frame to grayscale
gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


# # Detect corners to track in the first frame
prev_pts = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

total_yaw = 0

while index < 2592:
    index += 1

    print(index)

    img_path2 = "../imgdatapool/img" + str(index) + ".jpg"

    curr_frame = cv2.imread(img_path2)
    # Convert the current frame to grayscale
    gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    

    # Track points to the second frame
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, prev_pts, None)
    # print(f"This is type of status {type(status)}")
    if(type(status)==type(None)):
        print("Inside none condition")
        continue
    # print(f"This is type of status {type(status)}")
    while len(status)!=len(prev_pts):
        # print(f"This is length of status {len(status)}")
        # print(f"This is length of prev_pts {len(prev_pts)}")  # Threshold for re-detection
        # print("Inside loop here")
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, prev_pts, None)
        prev_pts = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # if len(prev_pts) < 10:  # Threshold for re-detection
       


    # Filter valid points using `status`
    status = status.flatten()  # Flatten `status`
    print(f"This is status {status}")
    print(f"This is prev_pts {prev_pts}")
    prev_pts = prev_pts[status == 1]
    
    curr_pts = curr_pts[status == 1]

    # Reshape points to the correct format for Fundamental Matrix computation
    prev_pts = prev_pts.reshape(-1, 1, 2)
    curr_pts = curr_pts.reshape(-1, 1, 2)

    # if prev_pts.shape[0] < 8 or curr_pts.shape[0] < 8:
        # print("Skipping frame due to insufficient points")
        # continue

    # Compute the Fundamental Matrix
    F, mask = cv2.findFundamentalMat(prev_pts, curr_pts, cv2.FM_RANSAC)
    
    if F is None or F.shape != (3, 3):
        print("Skipping frame due to invalid Fundamental Matrix")
        continue


    # # Compute the Fundamental Matrix
    # F, mask = cv2.findFundamentalMat(prev_pts, curr_pts, cv2.FM_RANSAC)

    # Compute the Essential Matrix
    E = K.T @ F @ K

    # Recover rotation and translation
    _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, K)

    yaw, pitch, roll = rotational_to_euler(R)
    
    print("Yaw = " ,yaw)

    prev_frame = copy.deepcopy(curr_frame)
    prev_pts = copy.deepcopy(curr_pts)

    total_yaw += yaw
    
    # Exit on ESC key
    if cv2.waitKey(45) & 0xFF == 27:
        break

print("Net Yaw: ", np.degrees(total_yaw))
cv2.destroyAllWindows()

