import cv2
import numpy as np
import os


def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Euler angles (yaw, pitch, roll)
    using a robust conversion method.
    """
    if np.isclose(np.abs(R[2, 0]), 1.0):
        # Handle singularity at poles
        pitch = np.pi / 2 * np.sign(R[2, 0])
        yaw = np.arctan2(R[0, 1], R[0, 2])
        roll = 0.0
    else:
        pitch = -np.arcsin(R[2, 0])
        cos_pitch = np.cos(pitch)
        roll = np.arctan2(R[2, 1] / cos_pitch, R[2, 2] / cos_pitch)
        yaw = np.arctan2(R[1, 0] / cos_pitch, R[0, 0] / cos_pitch)
    
    return yaw, pitch, roll


# Camera intrinsics matrix
K = np.array([
    [675.537322, 0.000000, 311.191300],
    [0.000000, 677.852071, 221.610964],
    [0, 0, 1]
])

# Algorithm parameters
FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# Initialize tracking
index = 2497
total_yaw = 0.0

# Load initial frame
img_path = f"./imgdatapool/img{index}.jpg"
prev_frame = cv2.imread(img_path)
if prev_frame is None:
    raise FileNotFoundError(f"Initial image not found: {img_path}")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)

# Main processing loop
for index in range(2498, 2593):  # Process images 2498-2592
    img_path = f"./imgdatapool/img{index}.jpg"
    curr_frame = cv2.imread(img_path)
    
    if curr_frame is None:
        print(f"Image not found: {img_path}")
        continue

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **LK_PARAMS
    )
    
    # Handle tracking failures
    if status is None or np.sum(status) < 8:
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
        prev_gray = curr_gray.copy()
        print(f"Tracking failed at {index}, re-detected features")
        continue
    
    # Filter valid points
    status = status.ravel()
    valid_prev = prev_pts[status == 1]
    valid_curr = curr_pts[status == 1]
    
    # Ensure sufficient points for pose estimation
    if len(valid_prev) < 8:
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
        prev_gray = curr_gray.copy()
        print(f"Insufficient points at {index}, re-detected features")
        continue
    
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(
        valid_prev, valid_curr, cv2.FM_RANSAC, 1.0, 0.99
    )
    
    # Check fundamental matrix validity
    if F is None or F.shape != (3, 3):
        print(f"Invalid fundamental matrix at {index}")
        prev_pts = valid_curr.reshape(-1, 1, 2)
        prev_gray = curr_gray.copy()
        continue
    
    # Compute essential matrix
    E = K.T @ F @ K
    
    # Recover pose
    _, R, t, _ = cv2.recoverPose(
        E, valid_prev, valid_curr, K, mask=mask
    )
    
    # Convert rotation and accumulate
    yaw, _, _ = rotation_matrix_to_euler(R)
    total_yaw += yaw
    
    # Prepare next iteration
    prev_gray = curr_gray.copy()
    prev_pts = valid_curr.reshape(-1, 1, 2)
    
    print(f"Frame {index}: ΔYaw = {np.degrees(yaw):.2f}°, Total = {np.degrees(total_yaw):.2f}°")
    

# Final output
print(f"\nNet Rotation: {np.degrees(total_yaw):.2f}°")